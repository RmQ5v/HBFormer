import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from timm.models.layers import drop_path


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
    
class Conv1d(nn.Module):
    default_act = nn.ReLU()
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True,norm=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.norm = nn.LayerNorm(c2) if norm is True else nn.Identity()      
        # self.norm = nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.conv(x).permute(0,2,1)
        return self.act(self.norm(x))

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x

# class Attention(nn.Module):
#     # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
#     def __init__(self, c, num_heads, proj_drop=0.):
#         super().__init__()
#         self.q1 = nn.Linear(c, c, bias=False)
#         self.k1 = nn.Linear(c, c, bias=False)
#         self.v1 = nn.Linear(c, c, bias=False)
#         self.ma1 = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
#         self.q2 = nn.Linear(c, c, bias=False)
#         self.k2 = nn.Linear(c, c, bias=False)
#         self.v2 = nn.Linear(c, c, bias=False)
#         self.ma2 = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        
#         self.proj = nn.Linear(c, c)
#         self.proj_drop = nn.Dropout(proj_drop)


#     def forward(self, x1, x2):
#         x1 = x1.permute(1,0,2)
#         x2 = x2.permute(1,0,2)
#         x1_x2 = self.ma1(self.q1(x1), self.k2(x2), self.v2(x2))[0].permute(1,0,2)
#         x2_x1 = self.ma2(self.q2(x2), self.k1(x1), self.v1(x1))[0].permute(1,0,2)
#         x = torch.cat((x1_x2, x2_x1), dim=1)
        
#         x = self.proj(x)
#         x = self.proj_drop(x)
        
#         return x
    
class Attention(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, proj_drop=0.):
        super().__init__()
        # self.q = nn.Linear(c, c, bias=False)
        # self.k = nn.Linear(c, c, bias=False)
        # self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.identity = nn.Identity()
        
        self.proj = nn.Linear(c, c)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        x = x.permute(1,0,2)
        # x = self.ma(self.q(x), self.k(x), self.v(x))[0].permute(1,0,2)
        (out, attn_map) = self.ma(x,x,x)
        x = out.permute(1,0,2)
        attn_map = self.identity(attn_map)
        # print(attn_map.shape)
        # x = self.ma(x, x, x)[0].permute(1,0,2)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

# class Attention(nn.Module):
#     def __init__(
#             self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
#             proj_drop=0., window_size=None, attn_head_dim=None):
#         super().__init__()
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         if attn_head_dim is not None:
#             head_dim = attn_head_dim
#         all_head_dim = head_dim * self.num_heads
#         self.scale = qk_scale or head_dim ** -0.5

#         self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
#         if qkv_bias:
#             self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
#             self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
#         else:
#             self.q_bias = None
#             self.v_bias = None

#         if window_size:
#             self.window_size = window_size
#             self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
#             # self.relative_position_bias_table = nn.Parameter(
#             #     torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
#             self.relative_position_bias_table = nn.Parameter(
#                 torch.zeros(num_heads, self.num_relative_distance))  # 2*Wh-1 * 2*Ww-1, nH
#             # cls to token & token 2 cls & cls to cls

#         else:
#             self.window_size = None
#             self.relative_position_bias_table = None
#             self.relative_position_index = None

#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(all_head_dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
        
#     def generate_2d_concatenated_self_attention_relative_positional_encoding_index(self, z_shape, x_shape):
#         '''
#             z_shape: (z_h, z_w)
#             x_shape: (x_h, x_w)
#         '''
#         z_2d_index_h, z_2d_index_w = torch.meshgrid(torch.arange(z_shape[0]), torch.arange(z_shape[1]))
#         x_2d_index_h, x_2d_index_w = torch.meshgrid(torch.arange(x_shape[0]), torch.arange(x_shape[1]))

#         z_2d_index_h = z_2d_index_h.flatten(0)
#         z_2d_index_w = z_2d_index_w.flatten(0)
#         x_2d_index_h = x_2d_index_h.flatten(0)
#         x_2d_index_w = x_2d_index_w.flatten(0)

#         concatenated_2d_index_h = torch.cat((z_2d_index_h, x_2d_index_h))
#         concatenated_2d_index_w = torch.cat((z_2d_index_w, x_2d_index_w))

#         diff_h = concatenated_2d_index_h[:, None] - concatenated_2d_index_h[None, :]
#         diff_w = concatenated_2d_index_w[:, None] - concatenated_2d_index_w[None, :]

#         z_len = z_shape[0] * z_shape[1]
#         x_len = x_shape[0] * x_shape[1]
#         a = torch.empty((z_len + x_len), dtype=torch.int64)
#         a[:z_len] = 0
#         a[z_len:] = 1
#         b=a[:, None].repeat(1, z_len + x_len)
#         c=a[None, :].repeat(z_len + x_len, 1)

#         diff = torch.stack((diff_h, diff_w, b, c), dim=-1)
#         _, indices = torch.unique(diff.view((z_len + x_len) * (z_len + x_len), 4), return_inverse=True, dim=0)
#         relative_position_index = torch.clamp(indices.view((z_len + x_len), (z_len + x_len)), 0 , self.num_relative_distance-1)
#         self.register_buffer("relative_position_index", relative_position_index)
        
#     def set_rel_pos_index(self,seq_len):
#         if self.window_size:
#             window_size = (1,seq_len)
#         # get pair-wise relative position index for each token inside the window
#             coords_h = torch.arange(window_size[0])
#             coords_w = torch.arange(window_size[1])
#             coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
#             coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
#             relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
#             relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
#             relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
#             relative_coords[:, :, 1] += window_size[1] - 1
#             relative_coords[:, :, 0] *= 2 * window_size[1] - 1
#             relative_position_index = \
#                 torch.zeros(size=(window_size[0] * window_size[1], ) * 2, dtype=relative_coords.dtype)
#             relative_position_index[:, :] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
#             relative_position_index = torch.clamp(relative_position_index, 0, self.num_relative_distance-1)
#             # relative_position_index[0, 0:] = self.num_relative_distance - 3
#             # relative_position_index[0:, 0] = self.num_relative_distance - 2
#             # relative_position_index[0, 0] = self.num_relative_distance - 1

#             self.register_buffer("relative_position_index", relative_position_index)

#     def forward(self, x, len_a, rel_pos_bias=None):
#         B, N, C = x.shape
#         # self.set_rel_pos_index(N)
#         len_b = N - len_a
#         self.generate_2d_concatenated_self_attention_relative_positional_encoding_index((1,len_a),(1,len_b))
#         qkv_bias = None
#         if self.q_bias is not None:
#             qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
#         # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
#         qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
#         qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

#         q = q * self.scale
#         attn = (q @ k.transpose(-2, -1))

#         if self.relative_position_bias_table is not None:
#             relative_position_bias = self.relative_position_bias_table[:, self.relative_position_index]
#             # relative_position_bias = \
#             #     self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # Wh*Ww,Wh*Ww,nH
#             # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
#             attn = attn + relative_position_bias.unsqueeze(0)

#         if rel_pos_bias is not None:
#             attn = attn + rel_pos_bias
        
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x


class block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.attn = Attention(
        #     dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)
        self.attn = Attention(dim, num_heads)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            # x = x + self.drop_path(self.attn(self.norm1(x), len_a, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.attn(self.norm1(x)))
            # x = self.norm1(x)
            # x = x + self.drop_path(self.attn(x[:,:len_a+1,:], x[:,len_a+1:,:]))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            # x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), len_a, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            # x = self.norm1(x)
            # x = x + self.drop_path(self.gamma_1 * self.attn(x[:,:len_a+1,:], x[:,len_a+1:,:]))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class OSI(nn.Module):
    def __init__(self, 
                 embed_dim,
                 hidden_dim,
                 num_heads, 
                 depth, 
                 mlp_ratio=4., 
                 qkv_bias=True, 
                 norm_layer=nn.LayerNorm, 
                 act_layer=nn.ReLU, 
                 window_size=(1,100), 
                 drop_path_rate=0.1,
                 attn_drop_rate=0.,
                 init_values=0.,
                 num_classes=1):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        if embed_dim != hidden_dim:
            self.conv = Conv1d(embed_dim, hidden_dim, 7, act=nn.Identity(), norm=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 201, hidden_dim) * .02)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block(
                dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size,
            )
            for i in range(depth)])
        
        self.norm = norm_layer(hidden_dim)
        self.fc_norm = norm_layer(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.num_heads = num_heads
        self.depth = depth
        
        self.initialize_weights()
        
        
    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m, fix_group_fanout=True):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            fan_out = m.kernel_size[0] * m.out_channels
            if fix_group_fanout:
                fan_out //= m.groups
            nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token','pos_embed'}
        
    def forward(self, seq_a, seq_b, add_a=None, add_b=None):
        outs = []
        batch_size = len(seq_a)
        for idx in range(batch_size):
            a = torch.from_numpy(seq_a[idx]).unsqueeze(0).cuda()
            b = torch.from_numpy(seq_b[idx]).unsqueeze(0).cuda()
            len_a = a.shape[1]
            len_b = b.shape[1]
            x = torch.cat((a,b),dim=1)
            if self.embed_dim != self.hidden_dim:
                x = self.conv(x)
            
            
            if add_a != None:
                cur_a = add_a[idx].cuda()
                cur_b = add_b[idx].cuda()
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), cur_a.expand(x.shape[0], -1, -1), x[:,:len_a,:], cur_b.expand(x.shape[0], -1, -1), x[:,len_a:,:]), dim=1)
                x = x + F.interpolate(self.pos_embed.permute(0,2,1),(len_a+len_b+3)).permute(0,2,1)
            else:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + F.interpolate(self.pos_embed.permute(0,2,1),(len_a+len_b+1)).permute(0,2,1)
            for idx, block in enumerate(self.blocks):
                # x = block(x, len_a)
                x = block(x)
            # if add_a != None:
            #     cur_a = add_a[idx].cuda()
            #     cur_b = add_b[idx].cuda()
            #     x = torch.cat((x[:,0,:], cur_a.expand(x.shape[0], -1, -1), x[:,1:len_a+1,:], cur_b.expand(x.shape[0], -1, -1), x[:,len_a+1:,:]), dim=1)
            x = self.norm(x)
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
            outs.append(self.head(x))
            # if add_a != None:
            #     outs.append(self.tran_block(a, b, add_a[idx].cuda(), add_b[idx].cuda()))
            # else:
            #     outs.append(self.tran_block(a, b))
        outs = torch.cat(outs)
        return outs

if __name__ == "__main__":
    # pos_embed = nn.Parameter(torch.randn(1, 101, 64) * .02)
    # x = F.interpolate(pos_embed.permute(0,2,1),(50)).permute(0,2,1)
    # print(x.shape)
    # import numpy as np
    # model = OSI(64, 64, 8, 1).cuda()
    # total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(total)
    # seq_a = np.random.randn(1,60,64).astype(np.float32)
    # seq_b = np.random.randn(1,30,64).astype(np.float32)
    # add_a = add_b = torch.randn((1,1,64)).cuda()
    # model(seq_a, seq_b, add_a, add_b)
    # model(seq_a,seq_b)
    model = transppi(64)
    a = torch.randn((64,150,20))
    b = torch.randn((64,250,20))
    model(a,b)

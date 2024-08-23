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
    
class Attention(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, proj_drop=0.):
        super().__init__()
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.identity = nn.Identity()
        
        self.proj = nn.Linear(c, c)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        x = x.permute(1,0,2)
        (out, attn_map) = self.ma(x,x,x)
        x = out.permute(1,0,2)
        attn_map = self.identity(attn_map)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
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
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
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
                x = block(x)
            x = self.norm(x)
            x = x[:, 1:].mean(dim=1)
            x = self.fc_norm(x)
            outs.append(self.head(x))
        outs = torch.cat(outs)
        return outs


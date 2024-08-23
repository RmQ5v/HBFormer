import torch
from .metrics import accuracy, auc, sensitivity, specificity, AUPRC, mcc_score, precision, recall, f1_score
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from timm.utils import accuracy
import util.utils as utils
import torch.distributed as dist

def train_one_epoch(model: torch.nn.Module,embedder,criterion,data_loader,
                    optimizer: torch.optim.Optimizer,loss_scaler,add_fea,args):
    model.train()
    total_iter = len(data_loader)
    print_interval = 5
    for iter, full_data in enumerate(data_loader):
        if add_fea:
            human_seq, virus_seq, human_pro_add, virus_pro_add, label = full_data
            label = label.cuda()
            with torch.no_grad():
                human_emb = list(embedder.embed_many(human_seq))
                virus_emb = list(embedder.embed_many(virus_seq))
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb,human_pro_add,virus_pro_add)
                loss = criterion(preds,label)
        else:           
            human_emb, virus_emb, label = full_data
            human_emb = human_emb.cuda().half()
            virus_emb = virus_emb.cuda().half()
            label = label.cuda()
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb)
                loss = criterion(preds,label)
            
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters(), clip_grad=0.1)
        if iter % print_interval == 0:
            cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
            print("{}/{} lr:{} loss:{}".format(iter, total_iter, cur_lr, loss.item()))
            

@torch.no_grad()            
def evaluate(model: torch.nn.Module,embedder,criterion,data_loader,add_fea,epoch,args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    preds_item_list = []
    labels_item_list = []
    for full_data in metric_logger.log_every(data_loader, 10, header):
        if add_fea:
            human_seq, virus_seq, human_pro_add, virus_pro_add, label = full_data
            label = label.cuda()
            human_emb = list(embedder.embed_many(human_seq))
            virus_emb = list(embedder.embed_many(virus_seq))
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb,human_pro_add,virus_pro_add)
                loss = criterion(preds,label)
        else:   
            human_emb, virus_emb, label = full_data
            label = label.cuda()
            human_emb = human_emb.cuda().half()
            virus_emb = virus_emb.cuda().half()
            with torch.cuda.amp.autocast():
                preds = model(human_emb,virus_emb)
                loss = criterion(preds,label)
        preds = torch.sigmoid(preds)
        preds_item_list.append(preds)
        labels_item_list.append(label)
        acc1 = accuracy(label.cpu().numpy().copy(), preds.cpu().numpy().copy())
        batch_size = label.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1, n=batch_size)
        
    # gather the stats from all processes
    preds_item_list = torch.cat(preds_item_list,dim=0)
    labels_item_list = torch.cat(labels_item_list,dim=0)
    all_item_list = torch.cat([preds_item_list,labels_item_list],dim=1)
    all_list = [None for _ in range(torch.distributed.get_world_size())]
    try:
        torch.cuda.empty_cache()
        dist.barrier()
        dist.all_gather_object(all_list, all_item_list)
    except:
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    preds_list = [item[:,0].cpu().numpy().copy() for item in all_list]
    labels_list = [item[:,1].cpu().numpy().copy() for item in all_list]
    preds_list = np.concatenate(preds_list,axis=0)
    labels_list = np.concatenate(labels_list,axis=0)
    #compute all the metrics
    acc_all = accuracy(labels_list,preds_list)
    auc_all = auc(labels_list,preds_list)
    auprc_all = AUPRC(labels_list, preds_list)
    pre_score_all = precision(labels_list,preds_list)
    re_score_all = recall(labels_list,preds_list)
    f1 = f1_score(labels_list,preds_list)
    mcc = mcc_score(labels_list,preds_list)
    sp = specificity(labels_list,preds_list)
    sn = sensitivity(labels_list, preds_list)
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    if args.output_dir and utils.is_main_process():
        output_dir = Path(args.output_dir)
        with (output_dir / "log.txt").open("a") as f:
            f.write("epoch:{:.4f} auc: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f} mcc:{:.4f} auprc:{:.4f} sp:{:.4f} sn:{:.4f} acc: {top1.global_avg:.4f} loss: {losses.global_avg:.4f} \n"
                    .format(epoch, auc_all, pre_score_all, re_score_all, f1, mcc, auprc_all, sp, sn, top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


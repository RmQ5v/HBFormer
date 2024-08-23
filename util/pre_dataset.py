import csv
import os
import numpy as np
import torch
import importlib
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# from gensim.models import KeyedVectors,word2vec,Word2Vec
# from bio_embeddings.embed import Word2VecEmbedder,ESM1bEmbedder, ProtTransBertBFDEmbedder
# from bio_embeddings.embed import ProtTransBertBFDEmbedder, BeplerEmbedder
from torch.utils.data import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import json
from sklearn.model_selection import StratifiedKFold
import scipy.io as scio
import pandas as pd
            
class Seq_Pair_dataset(Dataset):
    def __init__(self, csv_file, width, add_feature_file=None, return_pro_name=False):
        super().__init__()
        self.csv_file = csv_file
        self.len = 0
        self.width = width
        self.return_pro_name = return_pro_name
        self.human_seqs,self.virus_seqs,self.labels = [], [], []
        if add_feature_file != None:
            with open(add_feature_file,'r') as f:
                self.add_feature = json.load(f)
                f.close()
        if add_feature_file != None or self.return_pro_name:
            self.human_pro_list,self.virus_pro_list = [], []
        print("start read {} data".format(csv_file))
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.len += 1
                self.human_seqs.append(row['human_seq'])
                self.virus_seqs.append(row['virus_seq'])
                self.labels.append(float(row['labels']))
                if add_feature_file != None or self.return_pro_name:
                    self.human_pro_list.append(row["human_pro"].split(':')[-1].split('-')[0])
                    self.virus_pro_list.append(row["virus_pro"].split(':')[-1].split('-')[0])
            f.close()
        
        pos_weight = np.sum(self.labels.copy())
        neg_weight = len(self.labels) - pos_weight
        self.sample_weight = np.array(self.labels.copy())
        self.sample_weight[np.where(self.sample_weight==1.)] = 1. / pos_weight
        self.sample_weight[np.where(self.sample_weight==0.)] = 1. / neg_weight
        self.max_seq_len = 1000
                
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        if self.return_pro_name:
            human_pro_name = self.human_pro_list[index]
            virus_pro_name = self.virus_pro_list[index]
        if hasattr(self,"add_feature"):
            human_pro_name = self.human_pro_list[index]
            virus_pro_name = self.virus_pro_list[index]
            if human_pro_name in self.add_feature:
                human_pro_add = torch.FloatTensor([self.add_feature[human_pro_name]])
            else:
                human_pro_add = torch.zeros((1,self.width))
            if virus_pro_name in self.add_feature:
                virus_pro_add = torch.FloatTensor([self.add_feature[virus_pro_name]])
            else:
                virus_pro_add = torch.zeros((1,self.width))
        
        human_seq, virus_seq= self.human_seqs[index], self.virus_seqs[index]
        label = torch.FloatTensor([float(self.labels[index])])
        if hasattr(self,"add_feature"):
            if self.return_pro_name:
                return human_seq, virus_seq, human_pro_add, virus_pro_add, human_pro_name, virus_pro_name, label
            return human_seq, virus_seq, human_pro_add, virus_pro_add, label
        else:
            if self.return_pro_name:
                return human_seq, virus_seq, human_pro_name, virus_pro_name, label
            return human_seq, virus_seq, label
        

def get_dataloader(batch_size,fold_num,num_workers,width,add_file=None,train_idpd=None):
    if train_idpd:
        train_dataset = Seq_Pair_dataset("data_full/train_all.csv",width,add_file)
        test_dataset = Seq_Pair_dataset("data_full/independent.csv",width,add_file)
    else:
        train_dataset = Seq_Pair_dataset("data_full/{}/cv_train_{}.csv".format(fold_num,fold_num),width,add_file)
        test_dataset = Seq_Pair_dataset("data_full/{}/cv_test_{}.csv".format(fold_num,fold_num),width,add_file)
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset,batch_size,shuffle=False,num_workers=num_workers,pin_memory=True,sampler=train_sampler,prefetch_factor=6)
    test_loader = DataLoader(test_dataset,batch_size,shuffle=False,num_workers=num_workers,pin_memory=True,sampler=test_sampler,prefetch_factor=6)
    return train_loader, test_loader


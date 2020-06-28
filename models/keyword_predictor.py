# -*- coding: utf-8 -*-
# Name: keyword_predictor.py
# Author: Shao Shidong
# Date: 2020/5/26
# Version: Python 3.6 64-bit

import os
import sys
import torch
import numpy as np
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from utils.dataloader import SpiderDataset
from models.base_predictor import BasePredictor

class KeyWordPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(KeyWordPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.kw_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        # preprocess the num
        self.q_kw_num = ConditionalAttention(hidden_dim, use_bag_of_word=True)
        self.hs_kw_num = ConditionalAttention(hidden_dim, use_bag_of_word=True)
        self.kw_num_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 4))
        # preprocess the value
        self.q_kw = ConditionalAttention(hidden_dim, use_bag_of_word=False)
        self.hs_kw = ConditionalAttention(hidden_dim, use_bag_of_word=False)
        self.W_kw = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.kw_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.bce_logit = nn.BCEWithLogitsLoss(pos_weight=3*torch.tensor(3).cuda().double())

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len):
        q_enc,_ = self.q_lstm(q_emb_var, q_len)
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)
        kw_enc,_ = self.kw_lstm(kw_emb_var, kw_len)
        # predicting the num
        H_q_kw_num = self.q_kw_num(q_enc, kw_enc, q_len, kw_len) 
        H_hs_kw_num = self.hs_kw_num(hs_enc, kw_enc, hs_len, kw_len) 
        num_kw = self.kw_num_out(H_q_kw_num + int(self.use_hs)*H_hs_kw_num)
        # predicting the value
        H_q_kw = self.q_kw(q_enc, kw_enc, q_len, kw_len) 
        H_hs_kw = self.hs_kw(hs_enc, kw_enc, hs_len, kw_len) 
        H_kw = self.W_kw(kw_enc) 
        kw = self.kw_out(H_q_kw + int(self.use_hs)*H_hs_kw + H_kw).squeeze(2) 
        return (num_kw, kw)

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        batch_size = len(q_len)
        kw_emb_var, kw_len = embedding.get_history_emb(batch_size*[['where', 'order by', 'group by']])
        return self(q_emb_var, q_len, hs_emb_var, hs_len, kw_emb_var, kw_len)

    def loss(self, prediction, batch):
        loss = 0
        kw_num_score, kw_score = prediction
        kw_num_truth, kw_truth = batch['num_keywords'], batch['keywords']
        if not isinstance(kw_num_truth, torch.Tensor):
            kw_num_truth = torch.tensor(kw_num_truth).reshape(-1)     
        if not isinstance(kw_truth, torch.Tensor):
            kw_truth = torch.tensor(kw_truth).reshape(-1,3) 
        if len(kw_num_score.shape)<2:
            kw_num_score = kw_num_score.reshape(-1,4)
            kw_score = kw_score.reshape(-1,3)
        if kw_score.dtype != torch.float64:
            kw_score = kw_score.double()
            kw_num_score = kw_num_score.double()
        kw_num_truth = kw_num_truth.to(kw_num_score.device) 
        kw_truth = kw_truth.to(kw_score.device)

        loss += self.cross_entropy(kw_num_score, kw_num_truth)
        loss += self.bce_logit(kw_score, kw_truth) 
        return loss

    def accuracy(self, prediction, batch):     
        kw_num_score, kw_score = prediction
        kw_num_truth, kw_truth =  batch['num_keywords'], batch['keywords']
        batch_size = len(kw_truth)
        if not isinstance(kw_num_truth, torch.Tensor):
            kw_num_truth = torch.tensor(kw_num_truth).reshape(-1)       
        if not isinstance(kw_truth, torch.Tensor):
            kw_truth = torch.tensor(kw_truth).reshape(-1,3) 
        if len(kw_num_score.shape)<2:
            kw_num_score = kw_num_score.reshape(-1,4)
            kw_score = kw_score.reshape(-1,3)
        if kw_score.dtype != torch.float64:
            kw_score = kw_score.double()
            kw_num_score = kw_num_score.double()
        kw_num_truth = kw_num_truth.to(kw_num_score.device) 
        kw_truth = kw_truth.to(kw_score.device) 
        kw_num_prediction = torch.argmax(kw_num_score, dim=1)
        accuracy_num = (kw_num_prediction==kw_num_truth).sum().float()/batch_size
        correct_keywords = 0
        for i in range(batch_size):
            num_kw = kw_num_truth[i]
            correct_keywords += set(torch.argsort(-kw_truth[i,:])[:num_kw].cpu().numpy()) == set(torch.argsort(-kw_score[i,:])[:num_kw].cpu().numpy())
        accuracy_kw = correct_keywords/batch_size
        return accuracy_num.detach().cpu().numpy(), accuracy_kw

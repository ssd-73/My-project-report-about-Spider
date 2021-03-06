# -*- coding: utf-8 -*-
# Name: agg_predictor.py
# Author: Shao Shidong
# Date: 2020/5/26
# Version: Python 3.6 64-bit

import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    
from utils.lstm import PackedLSTM
from utils.attention import ConditionalAttention
from models.base_predictor import BasePredictor

class AggPredictor(BasePredictor):
    def __init__(self, num=6, *args, **kwargs):
        self.num = num
        super(AggPredictor, self).__init__(*args, **kwargs)
        
    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.col_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        # cmoputing aggregators
        self.q_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.hs_cs = ConditionalAttention(hidden_dim = hidden_dim, use_bag_of_word=True)
        self.W_cs = nn.Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.op_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, self.num))

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx):      
        batch_size = len(col_len)
        q_enc,_ = self.q_lstm(q_emb_var, q_len)
        hs_enc,_ = self.hs_lstm(hs_emb_var, hs_len)
        _, col_enc = self.col_lstm(col_emb_var, col_name_len) 
        col_enc = col_enc.reshape(batch_size, col_len.max(), self.hidden_dim) 
        col_emb = col_enc[np.arange(batch_size),col_idx].unsqueeze(1) 
        # predict aggregators
        H_q_cs = self.q_cs(q_enc, col_emb, q_len) 
        H_hs_cs = self.hs_cs(hs_enc, col_emb, hs_len) 
        H_cs = self.W_cs(col_emb).squeeze(1)   
        return self.op_out(H_q_cs + int(self.use_hs)*H_hs_cs + H_cs) 

    def predict(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx, force_agg = False):
        output = self.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx)
        if isinstance(output, tuple):
            numbers, values = output           
            numbers = torch.argmax(numbers, dim=-1).detach().cpu().numpy()
            predicted_values = []
            predicted_numbers = []
            for number,value in zip(numbers, values):
                if number>0:
                    predicted_values += [torch.argsort(-value)[:number].cpu().numpy()]
                predicted_numbers += [number]
            return (predicted_numbers, predicted_values)
        if force_agg:
            output[:,-1] = -math.inf
        return torch.argmax(output, dim=1).detach().cpu().numpy()

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])
        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape
        col_emb_var = col_emb_var.reshape(batch_size*num_cols_in_db, col_name_lens, embedding_dim) 
        col_name_len = col_name_len.reshape(-1)
        return self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, batch['column_idx'])

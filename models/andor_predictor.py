# -*- coding: utf-8 -*-
# Name: andor_predictor.py
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
from utils.attention import BagOfWord
from models.base_predictor import BasePredictor

class AndOrPredictor(BasePredictor):
    def __init__(self, *args, **kwargs):
        super(AndOrPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.bag_of_word = BagOfWord()
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_hs = nn.Linear(hidden_dim, hidden_dim)
        self.ao_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 2)) 

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len):
        q_enc, _ = self.q_lstm(q_emb_var, q_len)
        hs_enc, _ = self.hs_lstm(hs_emb_var, hs_len)
        # calculate H_Q
        H_Q = self.bag_of_word(q_enc, q_len)
        H_Q = self.W_q(H_Q)
        H_HS = self.bag_of_word(hs_enc, hs_len) 
        H_HS = self.W_hs(H_HS)
        return self.ao_out(H_Q + int(self.use_hs)*H_HS) 

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])    
        return self(q_emb_var, q_len, hs_emb_var, hs_len)

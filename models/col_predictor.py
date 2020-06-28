# -*- coding: utf-8 -*-
# Name: col_predictor.py
# Author: Shao Shidong
# Date: 2020/5/26
# Version: Python 3.6 64-bit

import os
import sys
import math
import torch
import numpy as np
import torch.nn as nn
from models.base_predictor import BasePredictor
from torch.utils.data import DataLoader
from embedding.embeddings import GloveEmbedding
from utils.dataloader import SpiderDataset, try_tensor_collate_fn
from utils.utils import length_to_mask
from utils.attention import ConditionalAttention
from utils.lstm import PackedLSTM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ColPredictor(BasePredictor):
    def __init__(self, max_num_cols=6, *args, **kwargs):
        self.num = max_num_cols
        super(ColPredictor, self).__init__(*args, **kwargs)

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        self.col_pad_token = -10000
        self.q_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.hs_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.col_lstm = PackedLSTM(input_size=N_word, hidden_size=hidden_dim//2, num_layers=num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.q_col_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)
        self.hs_col_num = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=True)
        self.col_num_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, self.num))  # num of cols: 1-6
        # predicting number of columns
        self.col_rep_out = nn.Sequential(
            nn.Tanh(), nn.Linear(hidden_dim, 4))  # num of repeats: 0-3
        # columns
        self.q_col = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.hs_col = ConditionalAttention(hidden_dim=hidden_dim, use_bag_of_word=False)
        self.W_col = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.col_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_dim, 1))
        pos_weight = torch.tensor(3).double()
        if gpu:
            pos_weight = pos_weight.cuda()
        self.bce_logit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len):
        batch_size = len(col_len)
        q_enc, _ = self.q_lstm(q_emb_var, q_len)
        hs_enc, _ = self.hs_lstm(hs_emb_var, hs_len)
        _, col_enc = self.col_lstm(col_emb_var, col_name_len)
        col_enc = col_enc.reshape(batch_size, col_len.max(), self.hidden_dim)
        # predict number of columns
        H_q_col = self.q_col_num(q_enc, col_enc, q_len, col_len)
        H_hs_col = self.hs_col_num(hs_enc, col_enc, hs_len, col_len)
        num_cols = self.col_num_out(H_q_col + int(self.use_hs)*H_hs_col)
        num_reps = self.col_rep_out(H_q_col + int(self.use_hs)*H_hs_col)
        # predicting columns 
        H_q_col = self.q_col(q_enc, col_enc, q_len, col_len)
        H_hs_col = self.hs_col(hs_enc, col_enc, hs_len, col_len)
        H_col = self.W_col(col_enc)
        cols = self.col_out(H_q_col + int(self.use_hs) * H_hs_col + H_col).squeeze(2)
        col_mask = length_to_mask(col_len).squeeze(2).to(cols.device)
        cols = cols.masked_fill_(col_mask, self.col_pad_token)
        return num_cols, num_reps, cols

    def process_batch(self, batch, embedding):
        q_emb_var, q_len = embedding(batch['question'])
        hs_emb_var, hs_len = embedding.get_history_emb(batch['history'])
        batch_size = len(q_len)
        col_emb_var, col_len, col_name_len = embedding.get_columns_emb(batch['columns_all'])
        batch_size, num_cols_in_db, col_name_lens, embedding_dim = col_emb_var.shape
        col_emb_var = col_emb_var.reshape(
            batch_size*num_cols_in_db, col_name_lens, embedding_dim)
        col_name_len = col_name_len.reshape(-1)
        return self(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)

    def loss(self, prediction, batch):
        loss = 0
        col_num_score, col_rep_score, col_score = prediction
        col_num_truth, col_truth = batch['num_columns'], batch['columns']
        col_rep_truth = torch.max(batch['columns'], dim=1)[0]-1
        col_truth = (col_truth > 0).double()
        if not isinstance(col_num_truth, torch.Tensor):
            col_num_truth = torch.tensor(col_num_truth).reshape(-1)
        if not isinstance(col_truth, torch.Tensor):
            col_truth = torch.tensor(col_truth).reshape(-1, 3)
        if len(col_num_score.shape) < 2:
            col_num_score = col_num_score.reshape(-1, col_num_score.size(0))
            col_score = col_score.reshape(-1, col_score.size(0))
        if col_score.dtype != torch.float64:
            col_score = col_score.double()
            col_num_score = col_num_score.double()
            col_rep_score = col_rep_score.double()
        col_num_truth = col_num_truth.to(col_num_score.device)-1
        col_rep_truth = col_rep_truth.to(col_rep_score.device).long()
        col_truth = col_truth.to(col_score.device)
        mask = col_score != self.col_pad_token
        
        loss += self.cross_entropy(col_num_score, col_num_truth.squeeze(1))
        loss += self.cross_entropy(col_rep_score, col_rep_truth)
        loss += self.bce_logit(col_score[mask], col_truth[mask])
        return loss

    def accuracy(self, prediction, batch):
        col_num_score, col_rep_score, col_score = prediction
        col_num_truth, col_truth = batch['num_columns'], batch['columns']
        col_rep_truth = torch.max(batch['columns'], dim=1)[0]-1
        col_truth = (col_truth > 0).double()
        batch_size = len(col_truth)
        if not isinstance(col_num_truth, torch.Tensor):
            col_num_truth = torch.tensor(col_num_truth).reshape(-1)
        if not isinstance(col_truth, torch.Tensor):
            col_truth = torch.tensor(col_truth).reshape(-1, 3)
        if len(col_num_score.shape) < 2:
            col_num_score = col_num_score.reshape(-1, col_num_score.size(0))
            col_score = col_score.reshape(-1, col_score.size(0))
        if col_score.dtype != torch.float64:
            col_score = col_score.double()
            col_num_score = col_num_score.double()
            col_rep_score = col_rep_score.double()
        col_num_truth = col_num_truth.to(col_num_score.device)
        col_rep_truth = col_rep_truth.to(col_rep_score.device).long()
        col_truth = col_truth.to(col_score.device)
        kw_num_prediction = torch.argmax(col_num_score, dim=1)
        accuracy_num = (kw_num_prediction+1 == col_num_truth.squeeze(1)).sum().float()/batch_size
        kw_rep_prediction = torch.argmax(col_rep_score, dim=1)
        accuracy_rep = (kw_rep_prediction == col_rep_truth).sum().float()/batch_size
        correct_keywords = 0
        for i in range(batch_size):
            num_kw = int(col_num_truth[i])
            num_rep = int(kw_num_prediction[i])
            trgt = torch.argsort(-col_truth[i, :])[:num_kw].cpu().numpy()
            pred = torch.argsort(-col_score[i, :])[:num_kw].cpu().numpy()
            reps = min(num_rep, num_kw)
            if reps > 0:
                pred = np.insert(pred, 0, np.ones(reps)*pred[0])[:-reps]
            correct_keywords += set(pred) == set(trgt)
        accuracy_kw = correct_keywords/batch_size
        return accuracy_num.detach().cpu().numpy(), accuracy_rep.detach().cpu().numpy(), accuracy_kw

    def predict(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, exclude_idx = None):
        output = self.forward(q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len)
        if isinstance(output, tuple):
            numbers, reps, values = output
            numbers = torch.argmax(numbers, dim=1).detach().cpu().numpy() + 1
            reps = torch.argmax(reps, dim=1).detach().cpu().numpy()
            predicted_values = []
            predicted_numbers = []
            for number, value, rep in zip(numbers, values, reps):
                if len(numbers) == 1 and exclude_idx != None:
                    count=-1
                    for i in exclude_idx:
                        count+=i
                        value[count] = -math.inf
                pred = torch.argsort(-value)[:number].cpu().numpy()
                rep = min(rep, number)
                if rep > 0:
                    pred = np.insert(pred, 0, np.ones(rep)*pred[0])[:-rep]
                predicted_values += [pred]
                predicted_numbers += [number]
            return predicted_numbers, predicted_values
        return torch.argmax(output, dim=1).detach().cpu().numpy()

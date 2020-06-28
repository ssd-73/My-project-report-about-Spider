# -*- coding: utf-8 -*-
# Name: base_predictor.py
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
from models import model_list

class BasePredictor(nn.Module):
    def __init__(self, N_word, hidden_dim, num_layers, gpu=True, use_hs=True):
        super(BasePredictor, self).__init__()
        self.N_word = N_word
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gpu = gpu
        self.use_hs = use_hs
        self.name = model_list.models_inverse[self.__class__.__name__]
        self.cross_entropy = nn.CrossEntropyLoss()
        self.construct(N_word, hidden_dim, num_layers, gpu, use_hs)
        if gpu: self.cuda()

    def construct(self, N_word, hidden_dim, num_layers, gpu, use_hs):
        pass
    
    def forward(self, q_emb_var, q_len, hs_emb_var, hs_len, col_emb_var, col_len, col_name_len, col_idx):
        pass

    def process_batch(self, batch, embedding):
        pass

    def loss(self, prediction, batch):
        truth = batch[self.name].to(prediction.device).long()
        if len(prediction.shape) == 1:
            prediction = prediction.unsqueeze(0)
        truth = truth.squeeze(dim=-1)
        # return the cross entropy error/loss
        return self.cross_entropy(prediction, truth)

    def accuracy(self, prediction, batch):
        truth =  batch[self.name]
        batch_size = len(truth)
        truth = truth.to(prediction.device).squeeze(1).long()
        prediction = torch.argmax(prediction, dim=-1)
        # compute accuracy
        accuracy = (prediction==truth).sum().float()/batch_size    
        return accuracy.detach().cpu().numpy()

    def predict(self, *args):
        output = self.forward(*args)
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
        return torch.argmax(output, dim=1).detach().cpu().numpy()

    def load(self, file_path):
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
        else:
            raise FileNotFoundError("Couldn't load model from {}!".format(file_path))

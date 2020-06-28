# -*- coding: utf-8 -*-
# Name: attention.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import torch
from utils.utils import length_to_mask
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Softmax

class Attention(Module):
    def __init__(self):
        super(Attention,self).__init__()

    def forward(self, value, key=None, query=None, mask=None):
        if key is None:
            key = query
        if query is None:
            query = value
        attention_weights = self.score(query, key, mask)
        context = attention_weights.matmul(value)
        return context, attention_weights

class GeneralAttention(Attention):  
    def __init__(self, hidden_dim):
        super(GeneralAttention,self).__init__()
        self.hidden_dim = hidden_dim
        self.W = Linear(in_features = hidden_dim, out_features = hidden_dim)
        self.softmax = Softmax(dim=-1)

    def score(self, query, key, mask=None):
        key = key.permute(0,2,1)
        atten = self.W(query).matmul(key)
        minimum = -float('inf')
        if mask is not None:
            atten.masked_fill_(mask, minimum)
        atten = atten.permute(0,2,1)
        attention_weights = self.softmax(atten) 
        return attention_weights

class UniformAttention(Attention):
    def __init__(self):
        super(UniformAttention, self).__init__()
        self.softmax = Softmax(dim=-1)        

    def score(self, query, key=None, mask=None):
        batch_size, seq_len, hidden_dim = query.shape        
        atten = torch.ones(batch_size, seq_len, 1, device = query.device)
        minimum = -float('inf')
        if mask is not None:
            atten.masked_fill_(mask,minimum)
        atten = atten.permute(0,2,1)
        attention_weights = self.softmax(atten) 
        return attention_weights

class ConditionalAttention(Module):
    def __init__(self, hidden_dim, use_bag_of_word=False):
        super(ConditionalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = GeneralAttention(hidden_dim)
        self.bag_of_word = UniformAttention()
        self.W = Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.use_bag_of_word = use_bag_of_word

    def forward(self, variable, condition, variable_lengths=None, condition_lengths=None):
        if variable_lengths is None and condition_lengths is None:
            mask_var_cond = None
            mask_cond = None
        elif condition_lengths is None:
            mask_var_cond = length_to_mask(variable_lengths).to(variable.device)
            mask_cond = None
        else:
            mask_var_cond = length_to_mask(variable_lengths, condition_lengths).to(variable.device) 
            mask_cond = length_to_mask(condition_lengths).to(variable.device) 
        H_var_cond, _ = self.attention(variable, key=condition, mask=mask_var_cond)
        if self.use_bag_of_word:
            H_var_cond, _ = self.bag_of_word(H_var_cond, mask=mask_cond) 
            H_var_cond = H_var_cond.squeeze(1)
        H_var_cond = self.W(H_var_cond)
        return H_var_cond

class BagOfWord(Module):
    def __init__(self):
        super(BagOfWord, self).__init__()
        self.attention = UniformAttention()

    def forward(self, variable, lengths):
        mask = length_to_mask(lengths)
        mask = mask.to(variable.device)
        context, _ = self.attention(variable, mask=mask)
        context = context.squeeze(1) 
        return context

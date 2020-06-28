# -*- coding: utf-8 -*-
# Name: lstm.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import torch
import numpy as np
from torch.nn import Module
from torch.nn import LSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class PackedLSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0, bidirectional=False):
        super(PackedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.lstm = LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)

    def forward(self, sequence, lengths):
        batch_size, seq_len, embedding_dim = sequence.shape
        mask = lengths>0
        lengths_filtered = lengths[mask]
        sequence_filtered = sequence[torch.tensor(mask)]
        sort_perm = np.array(sorted(range(len(lengths_filtered)), key=lambda k:lengths_filtered[k], reverse=True))  
        sort_inp_len = lengths_filtered[sort_perm]
        sort_perm_inv = np.argsort(sort_perm)
        if sequence.is_cuda:
            sort_perm = torch.tensor(sort_perm).long().cuda()
            sort_perm_inv = torch.tensor(sort_perm_inv).long().cuda()
        lstm_inp = pack_padded_sequence(sequence_filtered[sort_perm],
                sort_inp_len, batch_first=True)
        sort_ret_s, _ = self.lstm(lstm_inp, None)
        ret_s = pad_packed_sequence(
                sort_ret_s, batch_first=True)[0][sort_perm_inv]   
        if ret_s.size(0) != batch_size:
            mask = torch.tensor(mask).bool()
            tmp = torch.zeros(batch_size, seq_len, self.hidden_size + self.hidden_size*self.bidirectional).to(ret_s.device)
            tmp[mask,:,:] = ret_s
            ret_s = tmp   
        index = torch.LongTensor(lengths).unsqueeze(1).unsqueeze(2).expand(batch_size,1,ret_s.size(2)).to(ret_s.device)
        last_state = ret_s.gather(1, torch.abs(index-1))    
        return ret_s, last_state

# -*- coding: utf-8 -*-
# Name: utils.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import os
import torch
import pickle
import shutil
import numpy as np
import matplotlib.pyplot as plt

def calculate_mask(lengths,max_len, batch_size, device=torch.device('cpu')):
    lengths = torch.tensor(lengths,dtype=torch.long)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) >= lengths.unsqueeze(1)   
    mask = mask.unsqueeze(2)
    return mask
	
def length_to_mask(lengths, lengths2=None, max_len=None, max_len2=None, device=torch.device('cpu')):
    # find the max length
    max_len = max_len or int(max(lengths))
    batch_size = len(lengths)	
    mask = calculate_mask(lengths,max_len, batch_size)
    if lengths2 is not None:
        # find the max length
        max_len2 = max_len2 or int(max(lengths2))		
        mask2 = calculate_mask(lengths2,max_len2, batch_size)
        mask2 = mask2.expand(batch_size, max_len2, max_len)
        mask2 = mask2.permute(0,2,1)
        mask = mask.expand(batch_size, max_len, max_len2) #[batch_size, max_len, max_len2]
        mask = mask & mask2
    mask = mask.bool() # to match the requirement of Python 3.6, convert mask to mask.bool()
    return mask
	
def pad(sentences, pad_token=0):
    lengths = [len(sentence) for sentence in sentences]
    max_len = max(lengths)
    padded = []
    for example in sentences:
        pads = max_len - len(example)
        padded.append(np.pad(example, ((0,pads)), 'constant', constant_values=pad_token))		
    padded = np.asarray(padded)
    lengths = np.asarray(lengths)	
    return padded, lengths

def text2int(textnum, numwords={}):
    if not numwords:
        units = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight","nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen","sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        for index, word in enumerate(units):
            numwords[word] = (1, index)
        for index, word in enumerate(tens):
            numwords[word] = (1, index * 10)
    ordinal_words = {'first':1, 'second':2, 'third':3, 'fourth':4, 'fifth':5, 'sixth':6, 'seventh':7, 'eighth':8, 'ninth':9, 'tenth':10, 'twelfth':12}
    ordinal_endings = [('ieth', 'y')]
    current = result = 0
    curstring = ""
    onnumber = False
    num_tokens=len(textnum.split())
    tokens=textnum.split()
    for word in tokens:
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
            current = current * scale + increment
            if scale > 100:
                result += current
                current = 0
            onnumber = True
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)
            if word not in numwords:
                if onnumber:
                    curstring += repr(result + current) + " "
                if num_tokens-1 == tokens.index(word):
                    curstring += word + ""  
                else:  
                    curstring += word + " "
                result = current = 0
                onnumber = False
            else:
                scale, increment = numwords[word]
                current = current * scale + increment
                if scale > 100:
                    result += current
                    current = 0
                onnumber = True
    if onnumber:
        curstring += repr(result + current)
    return curstring

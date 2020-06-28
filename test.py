#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Name: test.py
# Author: Shao Shidong
# Date: 2020/5/28
# Version: Python 3.6 64-bit

import numpy as np
from tqdm import tqdm
from sql.syntaxsql import SyntaxSQL
from utils.dataloader import SpiderDataset
from embedding.embeddings import GloveEmbedding

emb = GloveEmbedding(path='data/glove.6B.300d.txt')
spider = SpiderDataset(data_path='data/dev.json', tables_path='data/tables.json', exclude_keywords=['-', ' / ', ' + ']) 
syntax_sql = SyntaxSQL(embeddings=emb, N_word=emb.embedding_dim, hidden_dim=100, num_layers=2, gpu=True, num_augmentation=0)

corrects_components = {'select':[],'where':[],'groupby':[],'orderby':[],'having':[],'limit_value':[],'keywords':[]}
corrects = 0
for i in tqdm(range(len(spider))):
    sample = spider[i]
    predicted_sql = syntax_sql.GetSQL(sample['question'], sample['db'])
    results = predicted_sql.component_match(sample['sql'])
    for result, component in zip(results, corrects_components):
        if result is not None:
            corrects_components[component] += [int(result)]
    if predicted_sql == sample['sql']:
        corrects += 1
        
print("\n# Components #")
for component in corrects_components:
    print(f"{component:12} accuracy = {np.mean(corrects_components[component])*100:0.2f}%   loss = {(np.asarray(corrects_components[component])==0).sum()*100/len(spider):0.2f}%")
print("\n#    Total   #")
print(f"total        accuracy = {corrects*100/len(spider):0.2f}%")

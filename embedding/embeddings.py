# -*- coding: utf-8 -*-
# Name: embeddings.py
# Author: Shao Shidong
# Date: 2020/5/25
# Version: Python 3.6 64-bit

import os
import torch
import pickle
import numpy as np
from torch.nn import Embedding, Module
from nltk.tokenize import word_tokenize

try:
    from something.datascience.transforms.embeddings.sentence_embeddings.laser.laser import LaserSentenceEmbeddings
    from something.datascience.transforms.embeddings.word_embeddings.fasttext.fasttext import FastTextWrapper, FasttextTransform
    raffle_import = True
except:
    raffle_import = False

class PretrainedEmbedding(Module):
    def __init__(self, num_embeddings, embedding_dim, word2idx, vectors, trainable=False, use_column_cache=True, gpu=True, use_embedding=True):
        super(PretrainedEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.word2idx = word2idx
        self.vectors = vectors
        self.column_cache={}
        self.use_column_cache = use_column_cache
        self.gpu = gpu
        if use_embedding:
            self.embedding = Embedding(num_embeddings, embedding_dim, padding_idx=0)
            self.embedding.weight.data.copy_(torch.from_numpy(vectors))
            if not trainable: self.embedding.weight.requires_grad = False              
        if gpu: self.cuda()
        self.device = torch.device("cuda" if self.gpu else "cpu")

    def forward(self, sentences, mean_sequence=False):
        if not isinstance(sentences, list):
            sentences = [sentences]       
        # convert to lowercase words
        sentences = [str.lower(sentence) for sentence in sentences]
        batch_size = len(sentences)
        sentences_words = [word_tokenize(sentence) for sentence in sentences]
        lenghts = [len(sentence) for sentence in sentences_words]
        max_len = max(lenghts)
        # create a 0 padding token
        indicies = torch.zeros(batch_size, max_len).long().to(self.device)
        # convert tokens to indices
        for i, sentence in enumerate(sentences_words):
            for j, word in enumerate(sentence):
                indicies[i,j] = self.word2idx.get(word,0)
        word_embeddings = self.embedding(indicies)
        if mean_sequence:
            word_embeddings = torch.sum(word_embeddings,dim=1)/torch.tensor(lenghts).float().to(self.device)
        return word_embeddings, np.asarray(lenghts)

    def embed_token(self, token):
        embs, words = [], token.split()
        for word in words:
            emb_list=[]
            for element in word.split('_'):
                if element:
                    emb,_ = self(element, mean_sequence=True)
                    emb_list.append(emb)
            embs.append(torch.mean(torch.stack(emb_list), dim=0))
        return torch.mean(torch.stack(embs), dim=0)

    def get_history_emb(self, histories):
        batch_size = len(histories)
        lengths = [len(history) for history in histories]
        max_len = max(lengths)
        # create tensor to store the resulting embeddings
        embeddings = torch.zeros(batch_size, max_len, self.embedding_dim).to(self.device)
        for i,history in enumerate(histories):
            for j, token in enumerate(history):
                emb = self.embed_token(token)
                embeddings[i,j,:] = emb
        return embeddings, np.asarray(lengths)
    def get_columns_emb(self, columns):
        batch_size = len(columns)
        # get the number of columns in each database
        lengths = [len(column) for column in columns]
        # get the number of tokens for each column
        col_name_lengths = [[len(word) for word in column] for column in columns]
        max_len = max(lengths)
        # join the column tokens      
        columns_joined = [[' '.join(column) for column in columns_batch] for columns_batch in columns]
        # get the number of tokens in each column
        col_name_lengths = [[len(word_tokenize(column)) for column in columns_batch] for columns_batch in columns_joined]
        # get the maximum number of tokens for all columns
        max_col_name_len = max([max(col_name_len) for col_name_len in col_name_lengths])
        embeddings = torch.zeros(batch_size, max_len, max_col_name_len, self.embedding_dim).to(self.device)
        col_name_lengths = np.zeros((batch_size, max_len))

        for i, db in enumerate(columns_joined):
            if str(db) in self.column_cache:
                cached_emb, cached_lengths = self.column_cache[str(db)]
                if self.gpu: cached_emb = cached_emb.cuda()
                min_size1 = min(cached_emb.size(0), max_len)
                min_size2 = min(cached_emb.size(1), max_col_name_len)
                embeddings[i,:min_size1,:min_size2,:] = cached_emb[:min_size1,:min_size2,:]
                col_name_lengths[i,:min_size1] = np.minimum(cached_lengths, max_col_name_len)[:min_size1]
                continue
            for j, column in enumerate(db):
                emb,col_name_len = self(column)
                embeddings[i,j,:int(col_name_len),:] = emb
                col_name_lengths[i,j] = int(col_name_len)
            if self.use_column_cache:
                self.column_cache[str(db)] = (embeddings[i,:,:].detach().cpu(), col_name_lengths[i,:])
        return embeddings, np.asarray(lengths),col_name_lengths


class GloveEmbedding(PretrainedEmbedding):
    def __init__(self, path='data/glove.6B.300d.txt', trainable=False, use_column_cache=True, gpu=True, embedding_dim=300):              
        word2idx, vectors = {}, []
        with open(path,'r', encoding ="utf8") as f:
            for idx, linee in enumerate(f,1):
                line = linee.split()       
                token = line[0]
                vector = line[1:]                
                if len(vector)==embedding_dim and token not in word2idx:                     
                    word2idx[token] = idx
                    vectors += [np.asarray(vector,dtype=np.float)]            
            word2idx['<unknown>'] = 0
            vectors.insert(0, np.zeros(len(vectors[0])))
        vectors = np.asarray(vectors, dtype=np.float)
        super(GloveEmbedding, self).__init__(num_embeddings=len(word2idx), 
            embedding_dim=len(vectors[0]),word2idx=word2idx, 
            vectors=vectors, trainable=trainable, use_column_cache=use_column_cache, gpu=gpu)

if raffle_import:
    class FastTextEmbedding(PretrainedEmbedding):
        def __init__(self, language='english', use_column_cache=True, gpu=True):
            self.fast = FasttextTransform(language)
            super(FastTextEmbedding, self).__init__(num_embeddings=None,
                embedding_dim=300,
                word2idx=None,
                vectors = None,
                trainable=False,
                use_column_cache=use_column_cache,
                gpu=gpu)

    class LaserEmbedding(PretrainedEmbedding):
        def __init__(self, path='data/laser_cached_en.pkl', gpu=True):
            super(LaserEmbedding, self).__init__(num_embeddings=1, 
                embedding_dim=1024,
                word2idx={}, 
                vectors=[], 
                trainable=False, 
                use_column_cache=True, 
                gpu=gpu, use_embedding=False)
            self.embedder = LaserSentenceEmbeddings()
            try:
                with open(path, 'rb') as file:
                    self.word2idx, self.vectors = pickle.load(file)
            except FileNotFoundError:
                pass
        def save(self, path):
            with open(path,'wb') as f:
                pickle.dump((self.word2idx, self.vectors), f)

        def forward(self, sentences, mean_sequence=False, language='en'):
            if not isinstance(sentences, list):
                sentences = [sentences]
            batch_size = len(sentences)
            sentences = [str.lower(sentence) for sentence in sentences]
            sentences_words = [word_tokenize(sentence) for sentence in sentences]
            lengths = [len(sentence) for sentence in sentences_words]
            max_len = max(lengths)                       
            if not isinstance(self.vectors, list):
                self.vectors = list(self.vectors)                
            if mean_sequence:
                word_embeddings = torch.zeros(batch_size, self.embedding_dim)          
                for i, sentence in enumerate(sentences):
                    word_embeddings[i] = torch.tensor(self.embedder(sentence, method="sentence", language=language))
            else:
                word_embeddings = torch.zeros(batch_size, max_len, self.embedding_dim)
                for i, sentence in enumerate(sentences_words):
                    for j, word in enumerate(sentence):
                        if word not in self.word2idx:
                            self.word2idx[word] = len(self.word2idx)
                            self.vectors.append(torch.tensor(self.embedder(word, method="sentence", language=language)))
                self.num_embeddings=len(self.word2idx)
                for i, sentence in enumerate(sentences_words):
                    for j, word in enumerate(sentence):
                        word_embeddings[i,j] = self.vectors[self.word2idx[word]]
            if self.gpu: word_embeddings = word_embeddings.cuda()
            return  word_embeddings, np.asarray(lengths)

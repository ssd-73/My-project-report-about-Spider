# -*- coding: utf-8 -*-
# Name: dataloader.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import os
import re
import sys
import json
import torch
import numpy as np
from itertools import chain
from torch.utils.data import Dataset  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql.sql import SQLStatement, DataBase, SQL_KEYWORDS, SQL_COND_OPS, SQL_AGG, SQL_OPS, SQL_ORDERBY_OPS, SQL_DISTINCT_OP
from itertools import chain
from utils.utils import pad
from utils.utils import text2int
from nltk.tokenize import word_tokenize

def zero_pad(sequences):
    lengths = [len(seq) for seq in sequences]
    padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq[:end]
    return padded_seqs, lengths

class SpiderDataset(Dataset):
    def __init__(self, data_path, tables_path, exclude_keywords=[], debug=True, language='en'):	
        directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))		
        self.exclude_keywords = exclude_keywords
        self.data = []
        data = json.load(open(directory + '/' + data_path, 'r', encoding="utf8"))
        exclude_keywords_counts = {key: 0 for key in exclude_keywords}
        for d in data:
            keywords = [keyword for keyword in exclude_keywords if str.upper(keyword) in str.upper(d['query'])]
            if keywords:
                for keyword in keywords:
                    exclude_keywords_counts[keyword] += 1
            else:
                self.data += [d]
        if debug:
            for keyword in exclude_keywords_counts:
                print(f"{exclude_keywords_counts[keyword]} queries with excluded keyword {keyword} are founded")
            print(f"The total number of removed queries = {len(data) - len(self.data)} / {len(data)}")
        tables = json.load(open(directory + '/' + tables_path, 'r'))
        # change the key of the dictionary
        self.tables = {}
        for table in tables:
            db_id = table['db_id']
            self.tables[db_id] = table
        self.samples = []
        p = re.compile(r"(?:(?<!\w)'((?:.|\n)+?'?)'(?!\w))")
        failed = 0
        for i in range(len(self.data)):
            try:
                example = self.data[i]
                db_id = example['db_id']
                db = DataBase(self.tables[db_id])
                sql = SQLStatement(query=example['query'], database=db)
                question = re.sub(p, u"\"\g<1>\"", example['question'][language])              
                history = sql.generate_history()
                sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                self.samples += [sample]
            except:
                failed += 1
        if failed > 0:
            print(f"{failed}/{len(self.data)} queries could not be loaded")
    def __len__(self):
        return len(self.data)
    def get_common_data(self, sample):
        db = sample['db']
        sql = sample['sql']
        # get a list of all columns in the database
        columns_all = db.to_list()
        # split connected words like 'address_id' into address, id
        columns_all_splitted = []
        for i, column in enumerate(columns_all):
            columns_tmp = []
            for word in column:
                columns_tmp.extend(word.split('_'))            
            columns_all_splitted += [columns_tmp]
        question = sample['question']
        return db, sql, columns_all_splitted, columns_all, question

    def __getitem__(self, idx):
        return self.samples[idx]

    def generate_keyword_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']
            # convert keywords to indices
            keywords_idx = [SQL_KEYWORDS.index(keyword) for keyword in sql.keywords]
            keywords_onehot = np.zeros(len(SQL_KEYWORDS))
            keywords_onehot[keywords_idx] = 1
            num_keywords = len(keywords_idx)
            history = sample['history']['keyword']
            question = sample['question']
            dataset.append({'num_keywords': num_keywords, 'keywords': keywords_onehot, 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='KeyWord')

    def generate_andor_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']
            question = sample['question']
            for andor, history in zip(sql.and_ors, sample['history']['andor']):
                andor_idx = SQL_COND_OPS.index(andor)                
                dataset.append({'andor': [andor_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='AndOr')

    def generate_column_dataset(self):
        dataset = []
        for sample in self.samples:
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            groups = [group for group in (sql.COLS, sql.WHERE, sql.GROUPBY, sql.HAVING, sql.ORDERBY) if group]
            for columns, history in zip(groups, sample['history']['col']):
                columns_idx = [columns_all.index(col.column.to_list()) for col in columns]
                # convert it to oneshot encoding
                columns_onehot = np.zeros(len(columns_all))
                for i in columns_idx:
                    columns_onehot[i]+= 1
                num_columns = len(columns_idx)
                dataset.append({'columns_all':columns_all_splitted, 'num_columns': [num_columns], 'columns': columns_onehot, 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Column')

    def generate_agg_dataset(self):
        dataset = []
        for sample in self.samples:           
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            columns = [group for group in chain(sql.COLS, sql.HAVING, sql.ORDERBY) if group]
            for column, history in zip(columns, sample['history']['agg']):
                column_idx = columns_all.index(column.column.to_list()) 
                # get index of the aggregator
                agg_idx = SQL_AGG.index(column.agg)
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'agg': [agg_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Aggregate')

    def generate_distinct_dataset(self):
        dataset = []
        for sample in self.samples:            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            columns = [group for group in chain(sql.COLS) if group]
            for column, history in zip(columns, sample['history']['distinct']):
                column_idx = columns_all.index(column.column.to_list()) 
                # get index of the aggregator
                dist_idx = SQL_DISTINCT_OP.index(column.distinct)
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'distinct': [dist_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Distinct')
    
    def generate_op_dataset(self):
        dataset = []
        for sample in self.samples:          
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            conditions = [group for group in chain(sql.WHERE, sql.HAVING) if group]
            for condition, history in zip(conditions, sample['history']['op']):
                column_idx = columns_all.index(condition.column.to_list()) 
                op_idx = SQL_OPS.index(condition.op) # get index of the aggregator
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'op': [op_idx], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='OP')

    def generate_having_dataset(self):
        dataset = []
        for sample in self.samples:
            db = sample['db']
            sql = sample['sql']           
            columns_all = db.to_list()
            columns_all_splitted = []
            for i, column in enumerate(columns_all):
                columns_tmp = []
                for word in column:
                    columns_tmp.extend(word.split('_'))                
                columns_all_splitted += [columns_tmp]
            column_idx = columns_all.index(sql.COLS[0].column.to_list()) 
            having = int(bool(sql.HAVING))
            history = sample['history']['having'][0]
            question = sample['question']
            dataset.append({'having': [having], 'column_idx':column_idx, 'columns_all':columns_all_splitted, 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Having')

    def generate_desasc_dataset(self):
        dataset = []
        for sample in self.samples:            
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)           
            for orderby, orderby_op, history in zip(sql.ORDERBY, sql.ORDERBY_OP, sample['history']['decasc']):
                column_idx = columns_all.index(orderby.column.to_list()) 
                desasc = SQL_ORDERBY_OPS.index(orderby_op)
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'desasc': [desasc], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Desasc')

    def generate_value_dataset(self):
        dataset = []
        for sample in self.samples:           
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)
            conditions = [group for group in chain(sql.WHERE, sql.HAVING) if group]
            for condition, history in zip(conditions, sample['history']['value']):
                column_idx = columns_all.index(condition.column.to_list())
                value = word_tokenize(str.lower(condition.value))
                if value:
                    start_token = value[0]
                    num_tokens = len(value)
                else:
                    start_token = ''
                    num_tokens = 1
                tokens=word_tokenize(text2int(str.lower(sample['question'])))
                values_onehot = np.zeros(len(tokens))
                try:
                    values_onehot[tokens.index(start_token)] = 1
                except:
                    pass               
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'value': values_onehot, 'num_tokens':[num_tokens], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='Value')

    def generate_limitvalue_dataset(self):
        dataset = []
        for sample in self.samples:           
            db, sql, columns_all_splitted, columns_all, question = self.get_common_data(sample)           
            for orderby, orderby_op, limitvalue, history in zip(sql.ORDERBY, sql.ORDERBY_OP, sql.LIMIT_VALUE, sample['history']['decasc']):
                column_idx = columns_all.index(orderby.column.to_list()) 
                desasc = SQL_ORDERBY_OPS.index(orderby_op)
                limitvalue = int(limitvalue)
                dataset.append({'columns_all':columns_all_splitted, 'column_idx': column_idx, 'desasc': [desasc], 'limitvalue': [limitvalue], 'question': question, 'history': history, 'db': db, 'sql': sql})
        return ModularDataset(dataset, name='LimitValue')

class ModularDataset(Dataset):
    def __init__(self, data, name=''):
        super(ModularDataset, self).__init__()
        self.data = data
        self.name = name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"{self.name}Dataset"


def try_tensor_collate_fn(batch):
    output = {}
    for example in batch:
        for key in example:
            if key in output:
                output[key] += [example[key]]
            else:
                output[key] = [example[key]]
    for key in output:
        try:
            output[key] = torch.tensor(pad(output[key])[0])
        except:
            pass
    return output

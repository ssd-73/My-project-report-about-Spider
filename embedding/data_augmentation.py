# -*- coding: utf-8 -*-
# Name: data_augmentation.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import os
import sys
import json
import random
import numpy as np
from torch.utils.data import Dataset  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sql.sql import SQLStatement, DataBase, SQL_KEYWORDS, SQL_COND_OPS, SQL_AGG, SQL_OPS, SQL_ORDERBY_OPS, SQL_DISTINCT_OP
from utils.dataloader import SpiderDataset

class AugmentedSpiderDataset(SpiderDataset):
    def __init__(self, data_path, tables_path, aug_data_path, aug_tables_path, exclude_keywords=[], debug=True, language='en', max_count=10000):
        directory=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        print(directory)	
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
                print(f"{exclude_keywords_counts[keyword]} queries with excluded keyword {keyword} have been found")
            print(f"The total number of removed queries = {len(data)-len(self.data)}/{len(data)}")
        self.tables = {}
        def add_tables_to_dict(path):
            tables = json.load(open(directory + '/' + path, 'r'))
            for table in tables:
                db_id = table['db_id']
                self.tables[db_id] = table
        add_tables_to_dict(aug_tables_path)
        num_aug_dbs = len(self.tables)
        add_tables_to_dict(tables_path)
        num_reg_dbs = len(self.tables) - num_aug_dbs
        print('Databases in augmentation set:', num_aug_dbs)
        print('Databases in regular training set:', num_reg_dbs)
        # generate augmented samples
        self.samples = []
        self.generate_augmented_samples(aug_data_path, max_count=max_count)
        # process regular training set
        failed = 0
        for i in range(len(self.data)):
            try:
                example = self.data[i]
                db_id = example['db_id']
                db = DataBase(self.tables[db_id])
                sql = SQLStatement(query=example['query'], database=db)
                question = example['question'][language]
                history = sql.generate_history()
                sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                self.samples += [sample]
            except:
                failed += 1
        if failed > 0:
            print(f"{failed}/{len(self.data)} queries could not be loaded")
        # shuffle to mix augmented samples with training set
        random.Random(1300135).shuffle(self.samples)

    def generate_augmented_samples(self, aug_data_path, max_count=10000):
        directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data = json.load(open(directory + aug_data_path, 'r', encoding="utf8"))
        count = 0
        if max_count == 0:
            print(f'Data augmentation has been disabled. Proceeding...')
            return
        # get the name of database
        for db_idx, db_name in enumerate(self.tables):        
            # generate database object
            db = DataBase(self.tables[db_name])
            # get the names of table
            table_names = self.tables[db_name]['table_names']
            table_names_original = self.tables[db_name]['table_names_original']
            # geet the names of et column
            col_idx = 0
            for table_idx, column_name in self.tables[db_name]['column_names']:
                # get the original column name
                column_name_original = self.tables[db_name]['column_names_original'][col_idx][1]
                col_idx += 1
                # generate the hash value
                hashval = hash(table_names[table_idx] + column_name_original + str(count)) # --> hash table

                if table_idx == -1 or hashval % 80 > 10:
                    continue
                for entry in data:
                    # get secondary and tertiary column names
                    num_columns = len(self.tables[db_name]['column_names']) - 1
                    j = (hashval + hash(col_idx + count)) % num_columns + 1
                    column_name_2 = self.tables[db_name]['column_names'][j][1]
                    column_name_original_2 = self.tables[db_name]['column_names_original'][j][1]
                    k = (j + hash(column_name_original) + count) % num_columns + 1
                    column_name_3 = self.tables[db_name]['column_names'][k][1]
                    column_name_original_3 = self.tables[db_name]['column_names_original'][k][1]
                    # generate values
                    value_int = ((hashval + j) % 10) + col_idx
                    value_int2 = ((hash(column_name_2) + j) % 10) + 1
                    for _ in range(3):
                        if hashval % 3 > 1: value_int *= 10
                        if (hashval + j) % 5 > 2: value_int2 *= 10
                    value_str = ['history','original','awarded','gene','approved','green','ivory','context','camp','qualified','peter','dependent','2012','parked','tent','paint','similar','persistent','couragous','twitch','dragon','court','green','purple','apple','samsung','fail','none','excellent','good','miss','mister','uncle','mother','eagle','great','fish','piano','fast','giant','jump','dive','doubt','match','disk','copy','calf','axis','soap','plot','coat','gap','hall','lost','cold','bet','fire','few','kit','stop']
                    value_str = value_str[j % len(value_str)]
                    # generate SQL query
                    query = entry['query'].replace('{COLUMN}', column_name_original).replace('{COLUMN2}', column_name_original_2).replace('{COLUMN3}', column_name_original_3).replace('{TABLE}', table_names_original[table_idx]).replace('{VALUE_INT}', str(value_int)).replace('{VALUE_INT2}', str(value_int2)).replace('{VALUE_STR}', value_str)
                    if (hashval + hash(query) + hash(count)) % 150 > 10: continue
                    try:
                        sql = SQLStatement(query=query, database=db)
                        history = sql.generate_history()
                        for question in entry['question']:
                            if (hashval + hash(question) + count) % 3 > 1:
                                continue
                            question = question.replace('{COLUMN}', column_name).replace('{COLUMN2}', column_name_2).replace('{COLUMN3}', column_name_3).replace('{TABLE}', table_names[table_idx]).replace('{VALUE_INT}', str(value_int)).replace('{VALUE_INT2}', str(value_int2)).replace('{VALUE_STR}', value_str)
                            sample = {'sql': sql, 'question': question, 'db': db, 'history': history}
                            self.samples += [sample]
                            count += 1
                            if count == max_count:
                                print(f'Generated {count} samples using {db_idx}/{len(self.tables)} databases. Proceeding...')
                                return                   
                    except:
                        continue
        print(f'{count} samples using all databases have been generated. Loading...')

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    spider = AugmentedSpiderDataset(data_path='data/train.json', tables_path='/data/tables.json', aug_data_path='/data/train_augment.json', aug_tables_path='/data/wikisql_tables.json', exclude_keywords=['-', ' / ', ' + '])
    dat = spider.generate_limitvalue_dataset()

# -*- coding: utf-8 -*-
# Name: sql.py
# Author: Shao Shidong
# Date: 2020/5/24
# Version: Python 3.6 64-bit

import re
import shlex
from itertools import zip_longest

SQL_OPS = ['=','>','<','>=','<=','!=','NOT LIKE','LIKE','BETWEEN']
SQL_AGG = ['max','min','count','sum','avg','']
SQL_COND_OPS = ['AND','OR','']
SQL_ORDERBY_OPS = ['DESC LIMIT','ASC LIMIT','DESC','ASC','LIMIT','']
SQL_DISTINCT_OP = ['distinct','']
SQL_KEYWORDS = ['where','group by','order by']

SQL_AGG_dict = {'max':'maximum', 'min':'minimum', 'count':'count','sum':'sum', 'avg':'average'}
SQL_ORDERBY_OPS_dict = {'DESC LIMIT':'descending limit', 'ASC LIMIT':'ascending limit', 'DESC':'descending', 'ASC':'ascending', 'LIMIT':'limit', 'NONE': 'none'}
SQL_OPS_dict = {'=':'=','>':'>','<':'<','>=':'> =','<=':'< =','!=':'! =','NOT LIKE':'not like','LIKE':'like','BETWEEN':'between'}

class SQLStatement():
    def __init__(self, query = None, database=None):
        self.COLS = []
        self.WHERE = []
        self.GROUPBY = []
        self.ORDERBY = []
        self.ORDERBY_OP = []
        self.TABLE = ""
        self.HAVING = []
        self.LIMIT_VALUE = ""
        self.database = database
        if isinstance(query, str):
            self.from_query(query)
    
    def __eq__(self, other):
        if not isinstance(other, SQLStatement):
            return NotImplemented
        if not self.TABLE :
            self.TABLE = self.COLS[0].column.table_name    
        if not other.TABLE :
            other.TABLE = other.COLS[0].column.table_name
        return (set(self.COLS)==set(other.COLS) and set(self.WHERE)==set(other.WHERE) and set(self.GROUPBY)==set(other.GROUPBY)and set(self.ORDERBY)==set(other.ORDERBY)and set(self.ORDERBY_OP)==set(other.ORDERBY_OP)
                and self.TABLE == other.TABLE and set(self.HAVING)==set(other.HAVING) and str(self.LIMIT_VALUE)==str(other.LIMIT_VALUE))

    def component_match(self, other):
        return ( 
            set(self.COLS)==set(other.COLS),
            set(self.WHERE)==set(other.WHERE) if self.WHERE else None,
            set(self.GROUPBY)==set(other.GROUPBY) if self.GROUPBY else None,
            set(self.ORDERBY)==set(other.ORDERBY) and set(self.ORDERBY_OP)==set(other.ORDERBY_OP) if self.ORDERBY else None,
            set(self.HAVING)==set(other.HAVING) if self.HAVING else None,
            str(self.LIMIT_VALUE)==str(other.LIMIT_VALUE) if self.LIMIT_VALUE else None,
            self.keywords == other.keywords)
         
    @property
    def keywords(self):
        keywords = []
        if self.WHERE:
            keywords += ['where']
        if self.GROUPBY:
            keywords += ['group by']
        if self.ORDERBY:
            keywords += ['order by']
        return keywords            
        
    @property
    def and_ors(self):
        andors = []
        for condition in self.WHERE:
            if condition.cond_op:
                andors += [condition.cond_op]
        for condition in self.HAVING:
            if condition.cond_op:
                andors += [condition.cond_op]
        return andors              

    def from_query(self, query):
        query = query.replace(';','')
        if ' - ' in query or ' + ' in query or ' / ' in query:
            raise NotImplementedError(f"Doesn't support arithmetic in query : {query}")
        if 'AS ' in query:
            aliases = re.findall(r'(?<=AS ).\w+', query)
            for alias in aliases:
                query = query.replace(f'{alias}.','')
                query = query.replace(f'AS {alias}','')
        clauses = ['SELECT','FROM','WHERE','GROUP BY','HAVING','ORDER BY']
        clauses = [clause for clause in clauses if clause in query]
        query_split = re.split(f'({"|".join(clauses)})',query)[1:]
        self.TABLE = str.lower(query_split[3]).strip()
        for i in range(0,len(query_split),2):
            clause = query_split[i]
            statement = query_split[i+1]
            if clause == 'SELECT':
                for column in statement.split(','):
                    agg, distinct = '',''
                    column = str.lower(column).strip()
                    if 'distinct(' in column:
                        distinct, column = column.split('(')
                        column = column.replace('(','').replace(')','')
                    if '(' in column:
                        agg = column.split('(')[0]
                        agg = agg.replace('distinct','').strip()
                        column = column.split('(')[1].split(')')[0].strip()
                    if 'distinct' in column:
                        distinct, column = column.split()                 
                    column = self.database.get_column(column, self.TABLE)
                    self.COLS.append(ColumnSelect(column, agg=agg, distinct=distinct))           
            elif clause == 'WHERE':
                if ' BETWEEN ' in statement:
                    col = statement.split(' BETWEEN ')
                    intervals = col[1].split(' AND ')
                    col = col[0].strip(' ')
                    statement = col + ' BETWEEN ' + intervals[0] +  ' ' + intervals[1]
                conditional_ops = re.findall('( AND | OR | BETWEEN )',statement)
                conditions = re.split(' AND | OR ',statement)
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):
                    if conditional_op == ' BETWEEN ':
                        column, op, value, valueless = condition.split(' ')
                        conditional_op = ''
                    else:
                        column, op, value = re.findall(r'(.*\(.*?\)|\'.*?\'|".*?"|NOT LIKE|\S+)',condition)
                        valueless = ""             
                    value = str.lower(value.strip('\'"%'))
                    valueless = str.lower(valueless.strip('\'"%'))
                    column = str.lower(column).strip()
                    conditional_op = conditional_op.strip()
                    column = self.database.get_column(column, self.TABLE)
                    self.WHERE.append(Condition(column, op, value, conditional_op,"","",valueless))
            elif clause == 'GROUP BY':
                for column in statement.split(','):
                    column = str.lower(column).replace('(','').replace(')','').strip()
                    column = self.database.get_column(column, self.TABLE)
                    column = ColumnSelect(column, '', '')
                    self.GROUPBY.append(column)
            elif clause == 'HAVING':
                if ' BETWEEN ' in statement:
                    col = statement.split(' BETWEEN ')
                    intervals = col[1].split(' AND ')
                    col = col[0].strip(' ')
                    statement = col + ' BETWEEN ' + intervals[0] +  ' ' + intervals[1]
                conditional_ops = re.findall('( AND | OR | BETWEEN )',statement)
                conditions = re.split(' AND | OR ',statement)
                for condition, conditional_op in zip_longest(conditions, conditional_ops, fillvalue=""):                 
                    if conditional_op == ' BETWEEN ':
                        column, op, value, valueless = condition.split(' ')
                        conditional_op = ''
                    else:
                        column, op, value = re.findall(r'(.*\(.*?\)|\'.*?\'|".*?"|NOT LIKE|\S+)',condition)
                        valueless=""
                    agg, distinct = '', ''
                    column = str.lower(column)
                    if '(' in column:
                        agg = column.split('(')[0].strip()
                        column = column.split('(')[1].split(')')[0]
                    if 'distinct' in column:
                        distinct, column = column.split()
                    value = str.lower(value.strip('\'"%'))
                    valueless = str.lower(valueless.strip('\'"%'))
                    conditional_op = conditional_op.strip()
                    column = self.database.get_column(column, self.TABLE)
                    self.HAVING.append(Condition(column, op, value, conditional_op, agg, distinct,valueless))
            elif clause == 'ORDER BY':
                for column in statement.split(','):
                    agg, distinct = '', ''
                    if '(' in column:
                        agg = column.split('(')[0].strip()                        
                        col = column.split('(')[1].split(')')[0].strip()                   
                    else:
                        col = column.split()[0].strip()
                    col = str.lower(col)
                    if 'distinct' in col:
                        distinct, col = col.split()
                    orderby_op = re.findall(f'({"|".join(SQL_ORDERBY_OPS[:-1])})', column)
                    if orderby_op:
                        self.ORDERBY_OP += orderby_op
                        if 'LIMIT' in orderby_op[0]:
                            fnd = re.findall(r'\d+', statement)
                            self.LIMIT_VALUE = fnd[-1]
                    else:
                        self.ORDERBY_OP += [""]
                    agg = str.lower(agg)
                    column = self.database.get_column(col, self.TABLE)
                    column = ColumnSelect(column, agg, distinct)
                    self.ORDERBY.append(column)
                                              
    def generate_history(self):
        history_dict = {'col':[], 'agg':[], 'andor':[], 'keyword':[], 'op':[], 'value':[], 'having':[], 'decasc':[], 'distinct':[]}
        history = ['select']
        history_dict['keyword'] += history.copy()
        history_dict['col'] += [history.copy()]
        for column in self.COLS:
            history += [' '.join(column.column.to_list())]
            history_dict['agg'] += [history.copy()]
            if column.agg:
                history += [column.agg]
            history_dict['col'] += [history.copy()]

            history_dict['distinct'] += [history.copy()]
            if column.distinct:
                history += [column.distinct]       
        if self.WHERE:
            history += ['where']           
            for condition in self.WHERE:
                history += [' '.join(condition.column.to_list())]   
                history_dict['op'] += [history.copy()]
                history += [condition.op]
                if condition.op == 'BETWEEN':
                    if condition.value:
                        history += [condition.value]
                    history_dict['value'] += [history.copy()]
                else:
                    history_dict['value'] += [history.copy()]
                    if condition.value:
                        history += [condition.value]
                history_dict['andor'] += [history.copy()]
                if condition.cond_op:                
                    history += [condition.cond_op]
            history_dict['col'] += [history.copy()]
        if self.GROUPBY:
            history += ['group by']
            for groupby in self.GROUPBY:
                history += [' '.join(groupby.column.to_list())]
            history_dict['col'] += [history.copy()]
        history_dict['having'] += [history.copy()]
        if self.HAVING:
            history += ['having']
            for condition in self.HAVING:
                history += [' '.join(condition.column.to_list())]       
                history_dict['agg'] += [history.copy()]
                if condition.column_select.agg:
                    history += [condition.column_select.agg]           
                history_dict['op'] += [history.copy()]
                history += [condition.op]
                history_dict['value'] += [history.copy()]
                if condition.value:
                    history += [condition.value]
                history_dict['andor'] += [history.copy()]
                if condition.cond_op:
                    history += [condition.cond_op]
            history_dict['col'] += [history.copy()]
        if self.ORDERBY:
            history += ['order by']
            for orderby in self.ORDERBY:
                history += [' '.join(orderby.column.to_list())]
                history_dict['agg'] += [history.copy()]
                if orderby.agg:
                    history += [orderby.agg]
            history_dict['col'] += [history.copy()]
        for orderby_op in self.ORDERBY_OP:
            history_dict['decasc'] += [history.copy()]                
            if orderby_op:
                history += [orderby_op]
        return history_dict

    def __str__(self):
        """Convert object to string representation of SQL"""
        string_cols = [str(col) for col in self.COLS]
        string_where = [str(where) for where in self.WHERE]
        string_group = [str(group) for group in self.GROUPBY]
        string_having = [str(having) for having in self.HAVING]
        string_order = [f"{str(order)} {orderop} {limitvalue}" for order, orderop, limitvalue in zip_longest(self.ORDERBY, self.ORDERBY_OP, [self.LIMIT_VALUE], fillvalue="") if self.ORDERBY]
        if not self.TABLE :
            self.TABLE = self.COLS[0].column.table_name
        sql_string = f"SELECT {','.join(string_cols)} FROM {self.TABLE}"
        if string_where:
            sql_string += f" WHERE {' '.join(string_where)}"
        if string_group:
            sql_string += f" GROUP BY {','.join(string_group)}"
        if string_having:
            sql_string += f" HAVING {' '.join(string_having)}"
        if string_order:
            sql_string += f" ORDER BY {','.join(string_order)}"
        sql_string = sql_string.replace('( ','(').replace('  ',' ')  
        return sql_string

class ColumnSelect():
    column = None
    agg = ""
    distinct = ""
    def __init__(self, column, agg="", distinct=""):
        self.column = column
        self.agg = agg
        self.distinct = distinct

    def __str__(self):
        assert self.column, "column name can't be empty"
        assert self.agg in SQL_AGG, f"{self.agg} is not an SQL aggregator"
        assert self.distinct in SQL_DISTINCT_OP, f"{self.distinct} is not an SQL distinct operator"
        if self.agg:
            return f"{self.agg}({self.distinct} {self.column})" 
        return f"{self.distinct} {self.column}"

    def __eq__(self, other):
        if not isinstance(other, ColumnSelect):
            return NotImplemented        
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))

class Condition():
    column_select = None
    op = ""
    value = ""
    cond_op = ""
    @property
    def column(self):
        return self.column_select.column
    
    @property
    def agg(self):
        return self.column_select.agg
    @agg.setter
    def agg(self, value):
        self.column_select.agg = value

    @property
    def distinct(self):
        return self.column_select.distinct
    
    @distinct.setter
    def distinct(self, value):
        self.column_select.distinct = value
        
    def __init__(self, column, op="", value="", cond_op="", agg="", distinct="",valueless=""):
        self.column_select = ColumnSelect(column, agg, distinct)
        self.op = op
        self.value = value
        self.cond_op = cond_op
        self.valueless = valueless

    def __str__(self):
        assert self.op in SQL_OPS, f"{self.op} is not an SQL operation"
        assert self.cond_op in SQL_COND_OPS, f"{self.cond_op} is not an SQL conditional operator (AND/OR)"

        if self.column.column_type == 'text':
            if "BETWEEN" in self.op:
                return f'{self.column_select} {self.op} "%{self.value}%" {SQL_COND_OPS[0]} {self.valueless}'
            elif "LIKE" in self.op:
                return f'{self.column_select} {self.op} "%{self.value}%" {self.cond_op}'
            else:
                return f'{self.column_select} {self.op} "{self.value}" {self.cond_op}'
        if "BETWEEN" in self.op:
            return f"{self.column_select} {self.op} {self.value} {SQL_COND_OPS[0]} {self.valueless}"        
        else:
            return f"{self.column_select} {self.op} {self.value} {self.cond_op}"
    
    def __eq__(self, other):
        if not isinstance(other, Condition):
            return NotImplemented       
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))
    
SQL_TYPES = ("text","number", "others")

class DataBase():
    def __init__(self, data):
        self.tables = []
        self.tables_dict = {}
        if isinstance(data, dict):
            self.from_dict(data)
    
    def from_dict(self, data_dict):
        self.db_name = data_dict['db_id']
        self.primary_keys = data_dict['primary_keys']
        columns_all = []
        column_types_all = []
        table_names = data_dict['table_names_original']      
        for (table_id, column_name), column_type in zip(data_dict['column_names_original'], data_dict['column_types']):
            column_name = str.lower(column_name)
            if table_id >= len(columns_all):
                columns_all.append([])
                column_types_all.append([])
            if table_id>=0:
                columns_all[table_id].append(column_name)
                column_types_all[table_id].append(column_type)

        for table_name, columns, column_types in zip(table_names, columns_all, column_types_all):
            table_name =  str.lower(table_name)
            columns.append('*')
            column_types.append('text')
            table = Table(table_name, columns, column_types)
            self.tables.append(table)
            self.tables_dict[table_name] = table

    def to_list(self):
        lst = []
        for table in self.tables:
            lst += [[col_type, table.table_name,col_name] for col_name, col_type in zip(table.columns, table.column_types)]
        return lst

    def get_column(self, column, table):
        return self.tables_dict[table].column_dict[column]

    def get_column_from_idx(self, idx):
        columns = []
        for table in self.tables:
            columns += list(table.column_dict.values())
        return columns[idx]

    def get_idx_from_column(self, column):
        columns = self.to_list()
        return columns.index(column.to_list())
        
class Table():
    def __init__(self, table_name, column_names, column_types):
        self.column_dict = {}
        for column_name, column_type in zip(column_names, column_types):
            col = Column(column_name, table_name, column_type)
            self.column_dict[column_name] = col
        self.table_name = table_name
        self.columns = column_names
        self.column_types = column_types

class Column():
    def __init__(self, column_name, table_name, column_type, database_name=""):
        self.column_name = column_name
        self.table_name = table_name
        self.column_type = column_type
        self.database_name = database_name
 
    def __str__(self):
        return self.column_name
    
    def __repr__(self):
        return str(self)

    def to_list(self):
        return [self.column_type, self.table_name,self.column_name]

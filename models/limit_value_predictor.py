# -*- coding: utf-8 -*-
# Name: limit_value_predictor.py
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
from models.agg_predictor import AggPredictor

class LimitValuePredictor(AggPredictor):
    def __init__(self, num=10, *args, **kwargs):
        super(LimitValuePredictor, self).__init__(*args, **kwargs, num=num)
        
    def predict(self, *args):
        return AggPredictor.predict(self, *args)

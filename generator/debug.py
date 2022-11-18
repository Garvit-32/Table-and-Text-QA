import os

import json
import ast
import numpy as np
import pandas as pd 
import evaluate
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, RobertaModel
import torch

list1 = ['21', '2', '123']
list1 = str(list1)
print(len(list1))
print(type(list1))
# output: 'It is a hall of worship ruled by Odin.'




# batch_size = 32
# nepochs = 20
# training_name = "t5-small-bs-32"
# model_checkpoint = "t5-base"



# print(model_checkpoint)
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# print(tokenizer.config)

# model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


# model = RobertaModel.from_pretrained('roberta-base')

# print(model.config)


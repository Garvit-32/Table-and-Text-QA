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



from transformers import AutoModelWithLMHead, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("tuner007/t5_abs_qa")
model = AutoModelWithLMHead.from_pretrained("tuner007/t5_abs_qa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_answer(question, context):
    # text = context
    # text = question + '</s>'  + context
    # input_text = f"context: {text}" 
    # input_text = f"context: {context}" 
    # input_text = f"context: {text}" 
    # input_text = "context: %s %s" % ()

    input_text = question + ' </s> '  + context
    print(input_text)
    
    features = tokenizer([input_text], return_tensors='pt')
    out = model.generate(input_ids=features['input_ids'].to(device), attention_mask=features['attention_mask'].to(device))
    
    return tokenizer.decode(out[0])



context = "In Norse mythology, Valhalla is a majestic, enormous hall located in Asgard, ruled over by the god Odin."
question = "What is Valhalla?"
print(get_answer(question, context))
# output: 'It is a hall of worship ruled by Odin.'


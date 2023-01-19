import ast
import torch 
import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from tqdm import tqdm
import numpy as np
import json
from utils import *


import warnings
warnings.filterwarnings('ignore')


import evaluate
metric = evaluate.load("squad_v2")


mode = 'dev'
dataset_path = 'dataset_tagop'
# suffix = 'io'
# table_csv_path = f'dataset_tagop/dev_{suffix}'
# model_checkpoint = "bert-large-bio-bs-16"
# model_checkpoint = "bert-large-num-bs-16"
# model_checkpoint = "bert-base-bs-32"
model_checkpoint = "roberta-ner-io-unfix-bs-16-epoch-30"
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
df = pd.DataFrame(columns = ['uid', 'order', 'question', 'ans', 'pred'])


token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="average", device = 0
)

def postprocess(uid, order, output, answer):
    text  = []

    # if len(output):
    #     if output[-1]['word'] ==  '.':
    #         output.pop()

    tmp = ''
    prev_end = 0

    for idx, i in enumerate(output):
        
        start = i['start']
        end = i['end']
        word = i['word']
        
        if prev_end + 1 == start:
            tmp += ' '

        elif prev_end != start and idx != 0:
            text.append(tmp)
            tmp = '' 

        tmp += word
        if idx == len(output) - 1:
            text.append(tmp)
        
        prev_end = end
    
    uid_order = uid + '_' + str(order)
    pred_dict = {'id' : uid_order}
    
    if len(text):
        text = list(map(str, text))
        text = [i.strip() for i in text]
        text.sort()
        text = ' '.join(text)
        pred_dict['no_answer_probability'] = 0.
    else:
        text = ''
        pred_dict['no_answer_probability'] = 1.


    pred_dict['prediction_text'] = text
    

    answer = list(map(str, answer))
    answer.sort()
    answer = ' '.join(answer)


    gt_dict = {'id' : uid_order}
    gt_dict['answers'] = {"text": [answer], 'answer_start':[0]}
    
    result = metric.compute(predictions=[pred_dict], references=[gt_dict])
    return result['exact'], result['f1'], text, answer

exact,f1 = [], []
   

for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    paras = ' '.join([para['text'] for para in paragraphs])
    # paras = paras.split(' ')
    # new_paras = []
    # for p in paras:
    #     num = to_number(p)
    #     if num is not None:
    #         new_paras.append(str(num))
    #     else:
    #         new_paras.append(p)
    
    # new_paras = ' '.join(new_paras)
    

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        mappings = ques['mapping']
      
        question = ques['question']

        if ques['answer_type'] != 'arithmetic' and ques['answer_from'] != 'table-text':

            text = question + ' </s> ' + paras
            # text = question + ' [SEP] ' + paras

            output = token_classifier(text)
            if ques['answer_from'] == 'table':
                ques['answer'] = []

            arg1, arg2, pred, ans = postprocess(uid, ques_order, output ,ques['answer'])
            # print(arg1, arg2)
            # if arg1 + arg2 == 0.0:
                # print(uid, ques_order)


            exact.append(arg1)
            f1.append(arg2)

            if arg1 == 0.0:

                row = {
                    'uid':uid,
                    'order': ques_order,
                    'question':question,
                    'ans': ans, 
                    'pred' : pred
                }
                df = df.append(row, ignore_index = True)

print(len(exact), len(f1))
print(np.mean(exact))
print(np.mean(f1))

# df.to_csv(f'result_bert.csv', index = False)

# 687d09ee-4aa8-402a-ae27-96ce063115a5 3
# eb122e2e-12b4-4e59-a018-7045192b021d 1
# eb122e2e-12b4-4e59-a018-7045192b021d 2



# 52.31481481481482
# 72.37284801924456

# old_data bs-16
# 55.401234567901234
# 76.32378104053979

# bert-large-bio-bs-16 train
# 69.08456167571761
# 85.44454493738296

# bert-large-bio-bs-32 train
# 68.79363847944143
# 85.40983374630115

# bert-large-bio-bs-16
# 53.858024691358025
# 75.49129442827005


# tweaks bert-large-num-bs-16
# 61.26543209876543
# 81.52669589300007

# bert-large-bio-bs-32
# 54.78395061728395
# 75.38143443644026


# bert-large-bs-16 dev   (io)
# 63.58024691358025
# 84.59619521164778

# bert-large-bs-32 dev   (io)
# 58.95061728395062
# 81.01094590625692


# roberta-io-fix bs 16
# 67.74691358024691
# 85.4354988519022

# roberta-ner-io-fix bs 16
# 69.44444444444444
# 85.57562370260585


# bert-ner-io-fix
# 53.24074074074074
# 72.66490412980602

# bert-large-bs-16-io-fix
# 53.39506172839506
# 74.47426248501118


# roberta-ner-io-unfix
# 70.67901234567901
# 86.21774660410246


# bert-ner-io-unfix
# 51.69753086419753
# 72.99904539020584
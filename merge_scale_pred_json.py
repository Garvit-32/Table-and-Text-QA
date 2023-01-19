# type: ignore
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'

import re
import ast
import json
import math
import torch 
import evaluate
from utils import to_number
import pandas as pd
from tqdm import tqdm
import numpy as np
from transformers import TapasForQuestionAnswering, TapasTokenizer, pipeline


import warnings
warnings.filterwarnings('ignore')

import evaluate
metric = evaluate.load("squad")


mode = 'dev'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

table_model_checkpoint = "tatqa_tapas/tapas_large"

text_model_checkpoint = "tatqa_text/roberta-ner-io-unfix"



op_model_checkpoint = "operator_pred/bert-tc-bs-42-hyp"



scale_model_checkpoint = "scale_pred/bert-tc-bs-42-scale"


gen_model_checkpoint = "expression_generator/t5-base-bs-16-clean"


table_csv_path = 'tatqa_tapas/dataset_tagop_table/dev'
tokenizer = TapasTokenizer.from_pretrained(table_model_checkpoint)
table_model = TapasForQuestionAnswering.from_pretrained(table_model_checkpoint)
table_model.to(device)

op_list = ["SPAN-TEXT", "SPAN-TABLE", "MULTI_SPAN", "CHANGE_RATIO",
                    "AVERAGE", "COUNT", "SUM", "DIFF", "TIMES", "DIVIDE" ,"MIXED"]

scale_list = ["", "thousand", "million",  "billion", "percent"]


exp_list = {
            1: ['-', '/', "(" , ")", '[', ']'],  # change ratio
            2: ['+', '/', "(" , ")", '[', ']'],  # average 
            4: ['+'], # sum
            5: ['-'], # diff
            6: ['*'], # times
            7: ['/'], # divide
            8: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
            }

text_model = pipeline(
    "token-classification", model=text_model_checkpoint, aggregation_strategy="average", device = 0
)


op_model = pipeline(
    "text-classification", model=op_model_checkpoint, device = 0
)

scale_model = pipeline(
    "text-classification", model=scale_model_checkpoint, device = 0, truncation= True
)

generator = pipeline(
    "translation_en_to_en", model=gen_model_checkpoint, device = 0, max_length = 128
)



EXCLUDE_IN_NUM = "'\"\\$€£¥%(),[]"
def _clean_num(text:str):
    return "".join([ch for ch in str(text) if ch not in EXCLUDE_IN_NUM])


def is_number(text: str) -> bool:
    try:
        words = " ".join([_clean_num(w) for w in text.split()]).split()
        # print(words)
        if len(words) == 0:
            """1023 or 1 million"""
            return False
        num = float(words[0])
        if np.isnan(num):
            return False
        if len(words) >= 2:
            if scale_to_num(words[1]) == 1:
                return False
        return True
    except ValueError:
        return False


def table_postprocess(uid, question):
    
    
    table = pd.read_csv(f'{table_csv_path}/{uid}.csv').astype(str) 
    encoding = tokenizer(table=table, 
                            queries= question, 
                            padding="max_length",
                            return_tensors="pt",
                            answer_coordinates=[], 
                            answer_text=[],
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    labels = encoding["labels"].float().to(device)

    

    outputs = table_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels = labels
    )

    predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(encoding, outputs.logits.detach().cpu())
    coordinates = predicted_answer_coordinates[0][0]


    if len(coordinates) == 0: 
        return []
            
    # only a single cell:
    elif len(coordinates) == 1:
        return [table.iat[coordinates[0]]]
        
    # multiple cells
    else:
        return [table.iat[coordinate] for coordinate in coordinates]


def op_postprocess(question):
    
    output = op_model(question)
    return int(output[0]['label'].split('_')[-1])
    

def text_postprocess(question, paragraphs):
    paras = ' '.join(paragraphs)

    text = question + ' </s> ' + paras
    
    output = text_model(text)
    

    text = []

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

    text = [" ".join(i.split()) for i in text]

    return text
    
    
def gen_postprocess(question, table_output, text_output, exp_list):

    operands = []

    for i in table_output:
        if isinstance(i, str):
            txt = i.split(' ')
            for j in txt:
                num = to_number(j)
                if num is not None:
                    operands.append(num)
        elif isinstance(i, (float, int)):
            operands.append(i)
            

    for i in text_output:
        txt = i.split(' ')
        for j in txt:
            num = to_number(j)
            if num is not None:
                operands.append(num)        

    operands.sort()
    operands = list(map(str, operands))

    inp = question + " </s> " +  " ".join(operands) + " </s> " + " ".join(exp_list)
    exp = generator(inp)
    
    return exp


def prepare_gt(uid, order, answers, scale):
    uid_order = uid + '_'+ str(order)
    gt_dict = {'id' : uid_order}


    if isinstance(answers,(float, int)):
        answer = str(answers)
    
    elif isinstance(answers,str):
        answer = answers

    elif isinstance(answers,list):
        answers.sort()
        answers = list(map(str, answers))
        answer = ' '.join(answers)
        answer = answer.lower()
    
    answer = answer[:-1] if answer[-1] == '.' else answer

    if len(scale):
        answer += f' {scale}'
        

    gt_dict['answers'] = {"text": [answer], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, table, text, ans, scale):

    pred_dict = {'id' : uid + '_'+ str(order)}
    # pred_dict['no_answer_probability'] = 1.

    if ans is not None:
        ans = ans[:-1] if ans[-1] == '.' else ans
        if len(scale):
            ans += f' {scale}'
        pred_dict['prediction_text'] = ans
        return pred_dict

    elif len(table) or len(text):
        table = np.unique(list(map(str, table)))
        text = np.unique(list(map(str, text)))
        answer = ''
             
        if len(table) != 0 and len(text) != 0:
            table.sort()
            text.sort()
            text_list = np.unique(np.concatenate((table,text)))
            answer = " ".join(text_list) 
            

        elif len(table) != 0:
            
            table.sort()
            answer = " ".join(table)

        
        elif len(text) != 0:
            text.sort()
            answer = " ".join(text)


        answer = answer[:-1] if answer[-1] == '.' else answer

        if len(scale):
            answer += f' {scale}'

        answer = answer.lower()
        pred_dict['prediction_text'] = answer

        return pred_dict

    else:
        pred_dict['prediction_text'] = ''
        if len(scale):
            pred_dict['prediction_text'] += f' {scale}'
        return pred_dict
         
        
def exp_postproces(exp, scale):
    try: 
        exp = re.sub('[!@%#$,]', '', str(exp))       
        ans = eval(exp)                    
        # if 'percentage' in question or '%' in table_output[0]:
        if scale == 'percent':
            ans *= 100
        ans = round(ans, 2)
        
        if isinstance(ans, float):
            if (ans).is_integer():
                ans = int(ans)

    except Exception:
        ans = 0

    return str(ans)


def scale_postprocess(text):
    
    output = scale_model(text)
    return scale_list[int(output[0]['label'].split('_')[-1])]
    


df = pd.DataFrame(columns = ['uid', 'order', 'type', 'from', 'question', 'ans', 'pred'])
data =json.load(open(f'operator_pred/dataset_tagop/tatqa_dataset_{mode}.json', 'r'))

exact, f1 = [],[]
pred = []
gt = []
ans_dict = {}

for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    text = [para['text'] for para in paragraphs]
    table = data[i]['table']
    table_data = table['table']


    para_text = [para['text'].strip() for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for a in table_data:
        for b in a:
            if len(b):
                table_tokens.append(b.strip())
    
    table_text = ' '.join(table_tokens)

    tmp_text = " [SEP] " + table_text + " [SEP] " + para_text  


    # if uid != '3ffd9053-a45d-491c-957a-1b2fa0af0570': continue

    questions = data[i]['questions']
    for ques in questions:
        ques_uid = ques['uid']
        ques_order = ques['order']
        mappings = ques['mapping']
        question = ques['question']
        answer = ques['answer']
        scale = ques['scale']
        ans = None

        # if ques_order != 1: continue

        gt_dict = prepare_gt(uid, ques_order, answer, scale)

        table_output = table_postprocess(uid, question)
        text_output = text_postprocess(question, text)
        operator = op_postprocess(question)
        pred_scale = scale_postprocess(ques['question'] + tmp_text)
        
        
        if operator > 0 :
            if operator == 3:  # For COUNT
                ans = str(len(table_output) + len(text_output))
            else:
                expression = gen_postprocess(question, table_output, text_output, exp_list[operator])
                exp = expression[0]['translation_text']
                ans = exp_postproces(exp, pred_scale)


        pred_dict = prepare_pred(uid, ques_order, table_output, text_output, ans, pred_scale)

        ans = pred_dict['prediction_text']

        if len(ans):
            tmp = ans.split(' ')
            if tmp[-1] not in ["thousand", "million", "billion", "percent"]:
                scale = ""
            else:
                scale = tmp[-1]
                ans = " ".join(tmp[:-1])
        else:
            scale = ""

        if len(ans):
            tmp = []
            flag = True
            for i in ans.split(" "):
                tmp.append(i)
                if not is_number(i):
                    flag = False
                    break
            if flag: ans = tmp
    
        ans_dict[ques_uid] = [ans, scale]

with open("eval_merge.json", "w") as outfile:
    json.dump(ans_dict, outfile)
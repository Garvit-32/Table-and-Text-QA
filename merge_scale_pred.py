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

# f = open(f"{mode}_log.txt", "a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

table_model_checkpoint = "tatqa_tapas/tapas_large"

text_model_checkpoint = "tatqa_text/roberta-ner-io-unfix"
# text_model_checkpoint = "tatqa_text/bert-large-bs-16"
# text_model_checkpoint = "tatqa_text/bert-large-num-bs-16"

op_model_checkpoint = "operator_pred/bert-tc-bs-42-hyp"
# op_model_checkpoint = "operator_pred/bert-tc-bs-42-arith"


scale_model_checkpoint = "scale_pred/bert-tc-bs-42-scale"
# op_model_checkpoint = "operator_pred/bert-tc-bs-42"

gen_model_checkpoint = "expression_generator/t5-base-bs-16-clean"
# gen_model_checkpoint = "expression_generator/t5-base-bs-32"

# gen_model_checkpoint = "expression_generator/t5-base-translation-bs-8"

table_csv_path = 'tatqa_tapas/dataset_tagop_table/dev'
tokenizer = TapasTokenizer.from_pretrained(table_model_checkpoint)
table_model = TapasForQuestionAnswering.from_pretrained(table_model_checkpoint)
table_model.to(device)

op_list = ["SPAN-TEXT", "SPAN-TABLE", "MULTI_SPAN", "CHANGE_RATIO",
                    "AVERAGE", "COUNT", "SUM", "DIFF", "TIMES", "DIVIDE" ,"MIXED"]

scale_list = ["", "thousand", "million",  "billion", "percent"]

# exp_list = {
#             3: ['-', '/', "(" , ")", '[', ']'],  # change ratio
#             4: ['+', '/', "(" , ")", '[', ']'],  # average 
#             6: ['+'], # sum
#             7: ['-'], # diff
#             8: ['*'], # times
#             9: ['/'], # divide
#             10: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
#             }

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
    # text = question + ' [SEP] ' + paras

    output = text_model(text)
    
    # if len(output):
    #     if output[-1]['word'] == '.':
    #         output.pop()
    # print(output)

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
    # print('='*60)
    # print(int(output[0]['label'].split('_')[-1]))
    # print('='*60)
    return scale_list[int(output[0]['label'].split('_')[-1])]
    

# count = []
# from collections import defaultdict
# store = defaultdict(list)

df = pd.DataFrame(columns = ['uid', 'order', 'type', 'from', 'question', 'ans', 'pred'])
data =json.load(open(f'operator_pred/dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
# f = open(f"log.txt", "w")
exact, f1 = [],[]
pred = []
gt = []

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
        pred.append(pred_dict)
        gt.append(gt_dict)
        
        # result = metric.compute(predictions=[pred_dict], references=[gt_dict])
        # arg1,arg2 =  result['exact_match'], result['f1']
        # return result['exact'], result['f1']
        # print(pred_dict)
        # print(gt_dict)
        # print(table_output)
        # print(text_output)
        # print(operator)
        # exact.append(arg1)
        # f1.append(arg2)
        
        if gt_dict['answers']['text'][0] !=  pred_dict['prediction_text']:
            
            row = {
                    'uid':uid,
                    'order': ques_order,
                    'type': ques['answer_type'],
                    'from': ques['answer_from'],
                    'question':question,
                    'ans': gt_dict['answers']['text'][0], 
                    'pred' : pred_dict['prediction_text']
            }

            df = df.append(row, ignore_index = True)
            

result = metric.compute(predictions=pred, references=gt)
print(result)
# print(np.mean(exact))
# print(np.mean(f1))

df.to_csv(f'results_merge_scale_nc.csv', index = False)
# f.close()


# 02913daf-213d-46e7-bf29-a65a8e64550f_4

        
        # # print(table_output)
        # # print(text_output)
        # # print('=' * 60)
        

# print(count)

        
        
# uid = '3ffd9053-a45d-491c-957a-1b2fa0af0570'
# question = 'What is the amount of total sales in 2019?'


# uid = '53474060-2736-46cb-bd97-1eb42f0ff3c1'
# question = 'How is industry end market information presented?'



        # if type(answer) not in count:
        #     count.append(type(answer))
        
        # if len(mappings.keys()) == 2: 
        #     if len(mappings['table']) != 0 and len(mappings['paragraph']) != 0 and ques['answer_type'] != 'arithmetic': 
        #         print(uid, ques_order)
        #         count.append(1)

        # if ques_order != 4: continue


# 4232c6c1-97cf-48ad-8b8b-f956871a3212 5
# 54c494f7-d731-49bf-b9cd-d494aea72e34 4
# cf49db0f-608a-4ef4-a248-d73c6030df4b 4
# b89656a2-196d-42d3-98bf-f58d51aedbb4 1
# 3661fba5-2876-41d7-9213-e86a6d5078dd 4
# 3661fba5-2876-41d7-9213-e86a6d5078dd 5
# 3661fba5-2876-41d7-9213-e86a6d5078dd 6



# O cases for text from both
# 7 cases in which operands from both table and text (arithmetic)

# old_exp
# 46.223021582733814 exact 
# 57.511846032284694 F1

# new_exp
# 47.302158273381295
# 58.590982722932175

# tweaks
# 58.333333333333336
# 69.83693790962951


# UID: 52164b70-6973-4844-af6a-76e8f1298d64
# ORDER: 4


# both
# {'exact_match': 54.13669064748201, 'f1': 66.41866513744009}

# with text bio-bs-16
# {'exact_match': 53.776978417266186, 'f1': 65.93776414394915}

# with text bio-bs-32
# {'exact_match': 54.49640287769784, 'f1': 66.2345706582395}



# with op

# text_model_checkpoint = "tatqa_text/bert-large-bs-16"
# op_model_checkpoint = "operator_pred/bert-tc-bs-42-arith"
# {'exact_match': 58.87290167865707, 'f1': 70.45967781294495}


# with scale
# {'exact_match': 57.13429256594724, 'f1': 76.83487358655381}

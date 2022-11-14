import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


import ast
import json
import math
import torch 
import evaluate
from utils import *
import pandas as pd
from tqdm import tqdm
from transformers import TapasForQuestionAnswering, TapasTokenizer, pipeline


import warnings
warnings.filterwarnings('ignore')

import evaluate
metric = evaluate.load("squad")


mode = 'dev'
data =json.load(open(f'operator_pred/dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
# f = open(f"{mode}_log.txt", "a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

table_model_checkpoint = "tatqa_tapas/tapas_large"
text_model_checkpoint = "tatqa_text/bert-large-bs-16"
op_model_checkpoint = "operator_pred/bert-tc-bs-42"
gen_model_checkpoint = "expression_generator/checkpoint-6500"

table_csv_path = 'tatqa_tapas/dataset_tagop_table/dev'
tokenizer = TapasTokenizer.from_pretrained(table_model_checkpoint)
table_model = TapasForQuestionAnswering.from_pretrained(table_model_checkpoint)
table_model.to(device)

op_list = ["SPAN-TEXT", "SPAN-TABLE", "MULTI_SPAN", "CHANGE_RATIO",
                    "AVERAGE", "COUNT", "SUM", "DIFF", "TIMES", "DIVIDE" ,"MIXED"]

exp_list = {
            3: ['-', '/', "(" , ")", '[', ']'],  # change ratio
            4: ['+', '/', "(" , ")", '[', ']'],  # average 
            6: ['+'], # sum
            7: ['-'], # diff
            8: ['*'], # times
            9: ['/'], # divide
            10: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
            }

text_model = pipeline(
    "token-classification", model=text_model_checkpoint, aggregation_strategy="average", device = 0
)


op_model = pipeline(
    "text-classification", model=op_model_checkpoint, device = 0
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
        num = to_number(text)
        if num is not None:
            return [num]
        else:
            return [text]
    
    # multiple cells
    else:
        cell_values = []
        for coordinate in coordinates:
            text = table.iat[coordinate]
            num = to_number(text)
            if num is not None:
                cell_values.append(num)
            else:
                cell_values.append(text)
                    
        return cell_values


def op_postprocess(question):
    
    output = op_model(question)
    return int(output[0]['label'].split('_')[-1])
    

def text_postprocess(question, paragraphs):
    paras = ' '.join(paragraphs)

    text = question + ' [SEP] ' + paras

    output = text_model(text)
    
    if len(output):
        if output[-1]['word'] == '.':
            output.pop()
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
        if isinstance(i, float) or  isinstance(i, int):
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


def prepare_gt(uid, order, answers):
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

    gt_dict['answers'] = {"text": [answer], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, table, text, ans):

    pred_dict = {'id' : uid + '_'+ str(order)}
    # pred_dict['no_answer_probability'] = 1.

    if ans is not None:
        pred_dict['prediction_text'] = ans
        return pred_dict

    elif len(table) or len(text):
        table = np.unique(list(map(str, table)))
        text = np.unique(list(map(str, text)))
            
        if len(table) != 0 and len(text) != 0:
            table.sort()
            text.sort()
            text_list = np.unique(np.concatenate((table,text)))
            pred_dict['prediction_text'] = " ".join(text_list)
            return pred_dict

        elif len(table) != 0:
            
            table.sort()
            pred_dict['prediction_text'] = " ".join(table)
            return pred_dict

        elif len(text) != 0:
            text.sort()
            pred_dict['prediction_text'] = " ".join(text)
            return pred_dict
    
    else:
        pred_dict['prediction_text'] = ''
        return pred_dict
         
            

# count = []
# from collections import defaultdict
# store = defaultdict(list)

exact, f1 = [],[]

for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    text = [para['text'] for para in paragraphs]

    # if uid != '02913daf-213d-46e7-bf29-a65a8e64550f': continue

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        mappings = ques['mapping']
        question = ques['question']
        answer = ques['answer']
        ans = None


        # if ques_order != 4: continue

        gt_dict = prepare_gt(uid, ques_order, answer)

        table_output = table_postprocess(uid, question)
        text_output = text_postprocess(question, text)
        operator = op_postprocess(question)
        
        if operator > 2 :

            if operator == 5:  # For COUNT
                ans = str(len(table_output) + len(text_output))
            else:
                expression = gen_postprocess(question, table_output, text_output, exp_list[operator])
                exp = expression[0]['translation_text']
                # print('Expression:', exp)
                try: 
                    ans = eval(str(exp))                    
                    ans = round(ans, 2)
                    if (ans).is_integer():
                        ans = int(ans)

                except Exception:
                    ans = 0
            
                ans = str(ans)

        pred_dict = prepare_pred(uid, ques_order, table_output, text_output, ans)
        result = metric.compute(predictions=[pred_dict], references=[gt_dict])
        arg1,arg2 =  result['exact_match'], result['f1']
        # return result['exact'], result['f1']
        exact.append(arg1)
        f1.append(arg2)

        # print('=' * 60)


print(np.mean(exact))
print(np.mean(f1))

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

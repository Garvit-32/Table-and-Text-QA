import ast
import torch 
import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import TapasTokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import TapasForQuestionAnswering, AdamW
from tqdm import tqdm
import numpy as np
# from utils import *
# import wandb

import evaluate
metric = evaluate.load("squad_v2")


model_checkpoint = 'tapas'
# model_checkpoint = 'tapas_large'
tokenizer = TapasTokenizer.from_pretrained(model_checkpoint)

# model = TapasForQuestionAnswering.from_pretrained('best_save_3')
model = TapasForQuestionAnswering.from_pretrained(model_checkpoint)
# model = TapasForQuestionAnswering.from_pretrained('tapas')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


df = pd.read_csv('dataset_tagop_table/dev.csv')
dataset_path = 'dataset_tagop_table'
table_csv_path = 'dataset_tagop_table/dev'

# print('='*60)
# print(dataset_path)
# print(table_csv_path)
# print('='*60)

def get_answers(uid, pred_answer_coord):

    pred_answers = []
    gt_answers = []

    for idx, coordinates in enumerate(pred_answer_coord[0]):
        table_uid = uid[idx].split('+')[0]
        question_position = int(uid[idx].split('+')[1])
        table = pd.read_csv(f'{dataset_path}/dev/{table_uid}.csv').astype(str)

        gt_dict = {'id' : uid[idx]}

        gt_answers_coords = ast.literal_eval(df[(df.uid == table_uid) & (df.position == question_position)].answer_cord.values[0])
        gt_answers_coords = set(gt_answers_coords)



        if len(gt_answers_coords) == 0: 

            gt_dict['answers'] = {"text": [''], 'answer_start':[]}

        else:
            cell_values = []
            for coordinate in gt_answers_coords:
                cell_values.append(table.iat[coordinate])
            
            cell_values = list(map(str, cell_values))
            cell_values.sort()
            gt_dict['answers'] = {"text": [", ".join(cell_values)], 'answer_start':[0]}


        gt_answers.append(gt_dict)
        pred_dict = {'id' : uid[idx]}
        if len(coordinates) == 0: 
            pred_dict['prediction_text'] = ''
            pred_dict['no_answer_probability'] = 1.
            
        elif len(coordinates) == 1:
            pred_dict['prediction_text'] = str(table.iat[coordinates[0]])
            pred_dict['no_answer_probability'] = 0.
        else:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            cell_values = list(map(str, cell_values))
            cell_values.sort()
            pred_dict['prediction_text'] = ", ".join(cell_values)
            pred_dict['no_answer_probability'] = 0.

        pred_answers.append(pred_dict)  
        ##############################################################


    return pred_answers, gt_answers


def get_inference(uid, position):


    item = df[(df.uid == uid) & (df.position == position)].values[0]

    table = pd.read_csv(f'{table_csv_path}/{uid}.csv').astype(str) # TapasTokenizer expects the table data to be text only
    encoding = tokenizer(table=table, 
                            queries=item[2], 
                            answer_coordinates=ast.literal_eval(item[3]), 
                            answer_text=ast.literal_eval(item[4]),
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding["token_type_ids"].to(device)
    labels = encoding["labels"].float().to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        labels=labels,
    )
    loss = outputs.loss

    # print(loss.item())
    
    uid = [uid + '+' + str(position)]
    predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(encoding, outputs.logits.detach().cpu(),)
    # cell_classification_threshold = 1.
    predicted_answers, gt_answers = get_answers(uid, predicted_answer_coordinates)
    # print(predicted_answers, gt_answers)

    
    
    result = metric.compute(predictions=predicted_answers, references=gt_answers)
    # print(result)
    # # print(result['f1'])
    # # print(result['exact'])

    # print(uid, result['exact'], result['f1'])
    return result['exact'], result['f1']


# '6bf238a5-0a3e-492d-91f8-7f62d3b37fba',1,Which countries does the group operate defined benefit schemes in?,[],[]
# '32edf644-acb0-4260-9392-f0baa4253f5a',2

exact,f1 = [], []

for i in tqdm(df.values):
    arg1, arg2 = get_inference(i[0], int(i[1]))
    exact.append(arg1)
    f1.append(arg2)

print(len(exact), len(f1))


print(np.mean(exact))
print(np.mean(f1))
# get_inference('6bf238a5-0a3e-492d-91f8-7f62d3b37fba',  0)

# 0fe00fcf-5d01-45b8-be3b-cc7faa0ddf08
# 0a75d1da-9beb-4a61-b2f4-06cff98b755e
# fdc2dbb8-0066-473e-95c1-43eb17223093
# f84f55c4-6ede-4bb6-9c24-49956f6e232a

# b3f4d2dd-a59b-45da-9608-e3401041a2b1,5

# tapas-base
# 51.91846522781775
# 86.24019003383849

# best_save_3
# 42.20623501199041
# 80.13849599172826

# tapas-large
# 74.76019184652279
# 89.20165191260489

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


import re
import ast
import evaluate
from utils import *
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

import evaluate
metric = evaluate.load("squad")


gen_model_checkpoint = "t5-base-bs-32"
table_csv_path = 'tatqa_tapas/dataset_tagop_table/dev'

generator = pipeline(
    "translation_en_to_en", model=gen_model_checkpoint, device = 0, max_length = 128
)

def prepare_gt(uid, order, answers):
    uid_order = uid + '_'+ str(order)
    gt_dict = {'id' : uid_order}
    try:
        ans = re.sub('[!@%#$,]', '', str(answers))       
        ans = re.sub('[[]', '(', ans)       
        ans = re.sub('[]]', ')', ans)       

        answer = eval(ans)
    except Exception:
        print(uid, order)
        exit(0)
    gt_dict['answers'] = {"text": [str(answer)], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, ans):

    pred_dict = {'id' : uid + '_'+ str(order)}
    pred_dict['prediction_text'] = str(ans)
    return pred_dict
     
        
def exp_postproces(exp):
    try: 
        exp = re.sub('[!@%#$,]', '', str(exp))       
        ans = eval(exp)                    

    except Exception:
        ans = 0

    return str(ans)

exact, f1 = [],[]
pred = []
gt = []

df = pd.read_csv('dataset_tagop/dev.csv')

for i in tqdm(df.values):

    gt_dict = prepare_gt(i[0], i[1], i[-1])
    
    expression = generator(i[2])
    exp = expression[0]['translation_text']
    ans = exp_postproces(exp)
    
    pred_dict = prepare_pred(i[0], i[1],  ans)
    pred.append(pred_dict)
    gt.append(gt_dict)


result = metric.compute(predictions=pred, references=gt)
print(result)
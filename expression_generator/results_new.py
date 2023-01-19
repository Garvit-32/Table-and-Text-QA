# type: ignore
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
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


# gen_model_checkpoint = "bart-base-bs-16-all"
# df = pd.read_csv('dataset_tagop/dev_all.csv')
gen_model_checkpoint = "bart-base-bs-32-old-hyp"
# gen_model_checkpoint = "bart-large-bs-32-not-op"
df = pd.read_csv('dataset_tagop/dev_not_op.csv')
max_length = 2048
save = pd.DataFrame(columns = ['uid', 'order','question', 'ans', 'pred'])

generator = pipeline(   
    "text2text-generation", model=gen_model_checkpoint, device = 0, max_length = max_length
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
        # b89656a2-196d-42d3-98bf-f58d51aedbb4
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



for idx,i in enumerate(tqdm(df.values)):

    gt_dict = prepare_gt(i[0], i[1], i[-1])
    

    expression = generator(i[-2].strip(), max_length = max_length, truncation = True)
    exp = expression[0]['generated_text']
    # exp = expression[0]['translation_text']

    ans = exp_postproces(exp)
    
    
    pred_dict = prepare_pred(i[0], i[1],  ans)
    pred.append(pred_dict)
    gt.append(gt_dict)


    if gt_dict['answers']['text'][0] !=  pred_dict['prediction_text']:

        row = {
                'uid':gt_dict['id'],
                'order': i[1],
                'question': i[-2].split('</s>')[0],
                'ans': i[-1], 
                'pred': exp
        }
        save = save.append(row, ignore_index = True)

    # if idx == 5:
    #     break

# print(pred, gt)


result = metric.compute(predictions=pred, references=gt)
print(result)
# save.to_csv('anal.csv', index=  False)

# {'exact_match': 64.99302649930264, 'f1': 64.99302649930264}
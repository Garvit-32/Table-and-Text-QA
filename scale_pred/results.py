# type: ignore
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'


import re
import ast
import evaluate
import pandas as pd
from tqdm import tqdm
from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

import evaluate
metric = evaluate.load("squad")


gen_model_checkpoint = "t5-base"
save = pd.DataFrame(columns = ['uid', 'order','question', 'ans', 'pred'])
df = pd.read_csv('dataset_tagop/dev.csv')

# gen_model_checkpoint = "t5-base-bs-32-only-num"

# generator = pipeline(
#     "text2text-generation", model=gen_model_checkpoint, device = 0 ,max_length = 1024
# )
generator = pipeline(
    "translation_en_to_en", model=gen_model_checkpoint, device = 0, max_length = 4096
)

def prepare_gt(uid, order, answers):
    uid_order = uid + '_'+ str(order)
    gt_dict = {'id' : uid_order}

    gt_dict['answers'] = {"text": [answers], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, ans):

    pred_dict = {'id' : uid + '_'+ str(order)}
    pred_dict['prediction_text'] = str(ans)
    return pred_dict
     
exact, f1 = [],[]
pred = []
gt = []


for i in tqdm(df.values):
    
    # if i[-3] == 'arithmetic': continue


    gt_dict = prepare_gt(i[0], i[1], i[-1])
    

    inputs = i[-2]
    # inputs = f'question: {i[-3]} context: {i[-2]}'
    
    # expression = generator(inputs, truncation = False)
    expression = generator(inputs, max_length = 4096)
    # print(expression)
    # ans = expression[0]['generated_text']
    ans = expression[0]['translation_text']
    
    pred_dict = prepare_pred(i[0], i[1],  ans)
    pred.append(pred_dict)
    gt.append(gt_dict)



    if gt_dict['answers']['text'][0] !=  pred_dict['prediction_text']:

        row = {
                'uid':gt_dict['id'],
                'order': i[1],
                'question':inputs.split('</s>')[0],
                'ans': gt_dict['answers']['text'][0], 
                'pred' : pred_dict['prediction_text']
        }
        save = save.append(row, ignore_index = True)
    

result = metric.compute(predictions=pred, references=gt)
save.to_csv('anal.csv', index=  False)
print(result)


# t5-base-bs-32-list-2-str    dev.csv    without scale contains both text and arithmetic
# {'exact_match': 30.455635491606714, 'f1': 40.93857234572136}


# t5-base-bs-32-scale-ml512-tT   _scale.csv    scale contains both text and arithmetic
# {'exact_match': 51.68421052631579, 'f1': 74.7734822775578}

# with arithmetic      same model but inference only on arithmetic prob     scale contains arithmetic
# {'exact_match': 29.73621103117506, 'f1': 59.46731104936841}
   

# t5-base-bs-32-derivation-scale-wo-text2text_derivation_scale_wo_text2text.csv    scale contains both text and arithmetic derivations
# {'exact_match': 50.65947242206235, 'f1': 67.19966222348955}


# t5-base-bs-32-only-span _text.csv    only text and scale
# {'exact_match': 59.78947368421053, 'f1': 79.96224232841088}

# t5-base-bs-32-only-text-count _count_text.csv    only text and scale with count
# {'exact_match': 60.31578947368421, 'f1': 79.11076007456583}

# t5-base-bs-32-only-num _arithmetic.csv   only arithmetic derivation and scale
# {'exact_match': 22.42339832869081, 'f1': 50.23393505287643}



























# t5-base-bs-32-scale-text2text  _text2text.csv    
# {'exact_match': 29.016786570743406, 'f1': 59.08421771774589}
# t5-base-bs-32-derivation-scale   t2t    _derivation_scale.csv
# model is predicting 2019 only
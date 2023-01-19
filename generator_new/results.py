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

# gen_model_checkpoint = "bart-base-bs-16-only-num-clean"
gen_model_checkpoint = "bart-large-bs-32-only-text-ml-512-epoch-15"
# gen_model_checkpoint = "t5-base-bs-16-only-num-clean"
df = pd.read_csv('dataset_tagop/dev_text.csv')
# df = pd.read_csv('dataset_tagop/dev_arithmetic.csv')
# gen_model_checkpoint = "t5-base-bs-32-only-num"

# generator = pipeline(
#     "text2text-generation", model=gen_model_checkpoint, device = 0 ,max_length = 1024
# )
save = pd.DataFrame(columns = ['uid', 'order','question', 'ans', 'pred'])
generator = pipeline(
    "text2text-generation", model=gen_model_checkpoint, device = 0, max_length = 4096, truncation = True
)

def prepare_gt(uid, order, answers):
    uid_order = uid + '_'+ str(order)
    gt_dict = {'id' : uid_order}
    answers = answers.lower()
    gt_dict['answers'] = {"text": [answers], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, ans):

    pred_dict = {'id' : uid + '_'+ str(order)}
    pred_dict['prediction_text'] = str(ans).lower()
    return pred_dict

def exp_postproces(exp):
    try: 
        scale = ''
        if any(c.isalpha() for c in exp):
            tmp = exp.split(' ')
            scale = tmp[-1]
            tmp.pop()
            exp = ' '.join(tmp)

        exp = re.sub('[!@%#$,]', '', str(exp))       
        ans = eval(exp)                    
        # if 'percentage' in question or '%' in table_output[0]:
        if scale == 'percent':
            ans *= 100
        ans = round(ans, 2)
        
        if isinstance(ans, float):
            if (ans).is_integer():
                ans = int(ans)

        if len(scale):
            ans = str(ans) + ' ' + scale
        else:
            ans = str(ans)

    except Exception:
        if len(scale):
            ans = scale
        else:
            ans = ''

    return ans
     
exact, f1 = [],[]
pred = []
gt = []


for i in tqdm(df.values):
    
    


    gt_dict = prepare_gt(i[0], i[1], i[-1].strip())
    # gt_dict = prepare_gt(i[0], i[1], exp_postproces(i[-1]))
    

    inputs = i[-2]

    expression = generator(inputs, max_length = 4096, truncation = True)
    ans = expression[0]['generated_text']
    # ans = exp_postproces(ans)
    
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
print(result)
# save.to_csv('anal_text_bart.csv', index=  False)



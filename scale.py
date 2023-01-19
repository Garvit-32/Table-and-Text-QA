# type: ignore
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
import re
import json
import evaluate
import pandas as pd
from tqdm import tqdm

from transformers import pipeline

import warnings
warnings.filterwarnings('ignore')

import evaluate
metric = evaluate.load("squad")



mode = 'dev'


op_model_checkpoint = "operator_pred/bert-tc-bs-42-hyp"
# op_model_checkpoint = "operator_pred/bert-tc-bs-42-arith"

# text_gen_model_checkpoint = "generator/t5-base-bs-32-only-span"
# arith_gen_model_checkpoint = "generator/t5-base-bs-32-only-num"
text_gen_model_checkpoint = "generator_new/bart-large-bs-16-only-text-ml-512-epoch-15"
arith_gen_model_checkpoint= "generator_new/bart-large-bs-16-only-num-ml-512-epoch-30"



exp_list = {
            1: ['-', '/', "(" , ")", '[', ']'],  # change ratio
            2: ['+', '/', "(" , ")", '[', ']'],  # average 
            4: ['+'], # sum
            5: ['-'], # diff
            6: ['*'], # times
            7: ['/'], # divide
            8: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
            }

op_model = pipeline(
    "text-classification", model=op_model_checkpoint, device = 0
)

# text_generator = pipeline(
#     "translation_en_to_en", model=text_gen_model_checkpoint, device = 0, max_length = 4096
# )
text_generator = pipeline(
    "text2text-generation", model=text_gen_model_checkpoint, device = 0, max_length = 4096, truncation = True
)

# arithmetic_generator = pipeline(
#     "translation_en_to_en", model=arith_gen_model_checkpoint, device = 0, max_length = 4096
# )
arithmetic_generator = pipeline(
    "text2text-generation", model=arith_gen_model_checkpoint, device = 0, max_length = 4096, truncation = True
)


table_csv_path = 'tatqa_tapas/dataset_tagop_table/dev'


df = pd.DataFrame(columns = ['uid', 'order', 'ques_uid',  'type', 'question', 'ans', 'pred'])
data =json.load(open(f'operator_pred/dataset_tagop/tatqa_dataset_{mode}.json', 'r'))
# f = open(f"log.txt", "w")
pred = []
gt = []


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


def op_postprocess(question):
    
    output = op_model(question)
    return int(output[0]['label'].split('_')[-1])


def prepare_gt(uid, order, answers, scale):
    uid_order = uid + '_'+ str(order)
    gt_dict = {'id' : uid_order}


    if isinstance(answers,(float, int)):
        answer = str(answers)
    
    elif isinstance(answers,str):
        answer = answers

    elif isinstance(answers,list):
        # answers.sort()
        answers = list(map(str, answers))
        answer = ' '.join(answers)

    if len(scale):
        answer = answer + ' ' + scale
    
    # answer = answer.lower()

    gt_dict['answers'] = {"text": [answer], 'answer_start':[0]}
    return gt_dict


def prepare_pred(uid, order, ans):
    # def prepare_pred(uid, order, table, text, ans):

    pred_dict = {'id' : uid + '_'+ str(order)}

    if ans is not None:
        pred_dict['prediction_text'] = ans
        return pred_dict

    else:
        pred_dict['prediction_text'] = ''
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


for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    table = data[i]['table']
    table_data = table['table']


    para_text = [para['text'].strip() for para in paragraphs]
    para_text = ' '.join(para_text)

    table_tokens = []
    for x in table_data:
        for y in x:
            if len(y):
                table_tokens.append(y.strip())
    
    table_text = ' '.join(table_tokens)

    text = " </s> " + table_text + " </s> " + para_text 

    # if uid != '52164b70-6973-4844-af6a-76e8f1298d64': continue

    questions = data[i]['questions']
    for ques in questions:
        ques_uid = ques['uid']
        ques_order = ques['order']
        question = ques['question']
        answer = ques['answer']
        scale = ques['scale']
        ans = None


        text_input = ques['question'] + text

        gt_dict = prepare_gt(uid, ques_order, answer, scale)

        operator = op_postprocess(question)
        

        if operator == 0:
            expression = text_generator(text_input, max_length = 4096)
            # ans = expression[0]['translation_text']
            ans = expression[0]['generated_text']

        elif operator == 3:
            expression = text_generator(text_input, max_length = 4096)
            # ans = expression[0]['translation_text']
            ans = expression[0]['generated_text']
            ans = str(len(ans.split('^')))
            
        else:
            expression = arithmetic_generator(text_input, max_length = 4096)
            ans = expression[0]['generated_text']
            # ans = expression[0]['translation_text']
            ans = exp_postproces(ans)

        # ans = ans.lower()
        pred_dict = prepare_pred(uid, ques_order, ans)

        pred.append(pred_dict)
        gt.append(gt_dict)
    
        if gt_dict['answers']['text'][0] !=  pred_dict['prediction_text']:

            row = {
                    'uid':uid,
                    'order': ques_order,
                    'ques_uid': ques_uid,
                    'type': ques['answer_type'],
                    'question':question,
                    'ans': gt_dict['answers']['text'][0], 
                    'pred' : pred_dict['prediction_text']
            }
            df = df.append(row, ignore_index = True)


    
result = metric.compute(predictions=pred, references=gt)
print(result)

df.to_csv(f'a.csv', index = False)



# scale count
# {'exact_match': 47.48201438848921, 'f1': 69.87476086073126}

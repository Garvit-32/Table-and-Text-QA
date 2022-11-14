import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from typing import List
from datautils import get_operator_class

import warnings
warnings.filterwarnings('ignore')

dataset = pd.DataFrame(columns = ['uid', 'order', 'text', 'derivation'])

OPERATOR_CLASSES = {
                    3: ['-', '/', "(" , ")", '[', ']'],  # change ratio
                    4: ['+', '/', "(" , ")", '[', ']'],  # average 
                    6: ['+'], # sum
                    7: ['-'], # diff
                    8: ['*'], # times
                    9: ['/'], # divide
                    10: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
                    }


mode = 'dev'
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))


count, total = 0, 0

for i in tqdm(range(len(data))):
    
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    questions = data[i]['questions']
    # text = [para['text'] for para in paragraphs]

    for ques in questions:
        if ques['answer_type'] == 'arithmetic':    
            ques_order = ques['order']  
                    
            operator = get_operator_class(
                ques['derivation'], 
                ques['answer_type'],
                ques['facts'],
                ques['answer'],
                ques['mapping']
                )

            derivation = re.sub('[!@%#$]', '', ques['derivation'])       

            question_text = ques['question'] + " </s> " + " ".join(ques['facts']) + " </s> " + " ".join(OPERATOR_CLASSES[operator])
        
            row = {
                'uid':uid,
                'order': ques_order,
                'text': question_text,
                'derivation': derivation, 
            }
            
            dataset = dataset.append(row, ignore_index = True)

dataset.to_csv(f'dataset_tagop/{mode}.csv', index = False)

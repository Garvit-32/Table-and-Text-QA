import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from typing import List
from datautils import get_operator_class
from utils import *

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
                    None: ['+', '-', '*', '/', "(" , ")", '[', ']'] # None
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

            derivation = re.sub('[!@%#$,]', '', ques['derivation'])       

            facts = ques['facts']
            new_facts = []
            for i in facts:
                num = to_number(i)
                if num is not None:
                    new_facts.append(num)
                else:
                    new_facts.append(i)
    
            ### This is not confirm will work or not 
            new_facts.sort()

            new_facts = list(map(str, new_facts))
            question_text = ques['question'] + " </s> " + " ".join(new_facts) + " </s> " + " ".join(OPERATOR_CLASSES[operator])
            # question_text = ques['question'] + " </s> " + " ".join(ques['facts']) + " </s> " + " ".join(OPERATOR_CLASSES[operator])
        
            row = {
                'uid':uid,
                'order': ques_order,
                'text': question_text,
                'derivation': derivation, 
            }
            
            dataset = dataset.append(row, ignore_index = True)

dataset.to_csv(f'dataset_tagop/{mode}.csv', index = False)

# c8ed0bf6-60c5-41e2-97e3-5ece54a1349b
# 7a9fdd23-2adc-4cf5-8761-5c7fbec53e6e 5 

# 53eec737-630e-4915-afbb-8c20cdd01263,6
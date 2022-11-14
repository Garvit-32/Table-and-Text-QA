import json
import numpy as np
import pandas as pd
from sklearn.decomposition import dict_learning 
from tqdm import tqdm
from pprint import pprint
from typing import List

import spacy
from spacy.training import offsets_to_biluo_tags
from spacy.tokenizer import Tokenizer
nlp = spacy.load('en_core_web_lg')
# nlp = spacy.load('en_core_web_sm')

import warnings
warnings.filterwarnings('ignore')

dataset = pd.DataFrame(columns = ['uid', 'order', 'question', 'text', 'answer', 'token', 'ner'])

# os.makedirs(f'dataset_tagop/{mode}', exist_ok = True)
mode = 'dev'
f = open(f"{mode}_log.txt", "a")
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))

tag_dict = {"B-ANS":1, 
            "U-ANS":1,
            "I-ANS":2,
            "L-ANS":2,  
            "O": 0
        }


def tag(text, enitites: List): 
    doc = nlp(text)

    tags = offsets_to_biluo_tags(doc,enitites)
    tokens = [i.text for i in doc]
    tags = [tag_dict[i] for i in tags]
    
    return tokens, tags

# tag("Who is Shaka Khan? Don't",[(7, 17, "ANS")])
for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    text = [para['text'] for para in paragraphs]
    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        tokens = []
        tags = []

        ques_token, ques_tag = tag(ques['question'], []) 
        tokens.extend(ques_token + ['[SEP]'])
        tags.extend(ques_tag + [None])
        
        para_order = []
        if 'paragraph' in ques['mapping'].keys():
            para_order = list(ques['mapping']['paragraph'].keys())

        for para in paragraphs:
            if str(para['order']) in para_order:
                mappings = ques['mapping']['paragraph'][str(para['order'])]
                # mappings = [(i[0],i[1], 'ANS') for i in mappings]
                entities = []
                for x,y in mappings:
                    if para['text'][x] == ' ': x += 1
                    # print(para['text'], x,y )
                    if y < len(para['text']) and para['text'][y - 1] == ' ': y-= 1
                    # if y < len(para['text']) and para['text'][y] == ' ':
                    #     if y + 1 < len(para['text']) and para['text'][y + 1] == ' ':
                    #         y += 1
                    #     elif para['text'][y - 1] == ' ':
                    #         y -= 1
                    entities.append((x,y, 'ANS'))
                     
            
                # print(para['text'], entities, i)
                try: 
                    para_token, para_tag = tag(para['text'], entities)
                except:
                    # if uid == 'daf81839-002f-40c2-8067-b4ad7eaf1517':
                    # else:
                    # print(f"UID: {uid}")
                    # print(f"ENTITIES: {entities}")
                
                    f.write('Continuing\n')
                    f.write(f"UID: {uid}\n")
                    f.write(f"ORDER: {ques_order}\n")
                    f.write(f"TEXT: {para['text']}\n")
                    f.write(f"MAPPING: {mappings}\n")
                    f.write(f"ENTITIES: {entities}\n")
                    f.write('=' * 60 + '\n')
                    # exit(0)
                    continue
                        # for x,y  in mappings:
                        #     print(, x, y)
                        
                        # exit(0)

                
                tokens.extend(para_token)
                tags.extend(para_tag)
            else: 
                para_token, para_tag = tag(para['text'], []) 
                tokens.extend(para_token)
                tags.extend(para_tag)

        row = {
            'uid':uid,
            'order': ques_order,
            'question':ques['question'],
            'text':' '.join(text),
            'answer': ques['answer'], 
            'token': tokens, 
            'ner': tags
        }

        dataset = dataset.append(row, ignore_index = True)
        # print(tokens)
        # print(tags)
        # print('=' * 60)


dataset.to_csv(f'dataset_tagop/{mode}.csv', index = False)
f.close()
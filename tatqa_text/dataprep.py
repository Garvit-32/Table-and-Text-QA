import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from utils import *
warnings.filterwarnings('ignore')

dataset = pd.DataFrame(columns = ['uid', 'order', 'question', 'text', 'answer', 'token', 'ner'])

mode = 'train'
suffix = 'roberta'
f = open(f"{mode}_{suffix}_log.txt", "a")
data =json.load(open(f'dataset_tagop/tatqa_dataset_{mode}.json', 'r'))


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def to_number(text:str) -> float:
    num = extract_one_num_from_str(text)
    scale_val = word_scale_handle(text)
    negative_flag = negative_num_handle(text)
    percent_flag = percent_num_handle(text)
    if num is not None:
        return round(num * scale_val * negative_flag * percent_flag, 4)
    return None

# def token_tags(text: )

def question_tags(text):
    tokens = []
    tags = []
    prev_is_whitespace = True
    current_tags = [0 for i in range(len(text))]
    start_index = 0
    wait_add = False
    for i, c in enumerate(text):
        if is_whitespace(c):  # or c in ["-", "–", "~"]:
            if wait_add:
                if 1 in current_tags[start_index:i]:
                    tags.append(1)
                else:
                    tags.append(0)
                wait_add = False
            prev_is_whitespace = True
        elif c in ["-", "–", "~"]:
            if wait_add:
                if 1 in current_tags[start_index:i]:
                    tags.append(1)
                else:
                    tags.append(0)
                wait_add = False
            tokens.append(c)
            tags.append(0)
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                tokens.append(c)
                wait_add = True
                start_index = i
            else:
                tokens[-1] += c
            prev_is_whitespace = False
    if wait_add:
        if 1 in current_tags[start_index:len(text)]:
            tags.append(1)
        else:
            tags.append(0)
    
    return tokens, tags 

def paragraph_tags(paragraphs, mapping):
    # mapping_content = []
    paragraphs_copy = paragraphs.copy()
    paragraphs = {}
    for paragraph in paragraphs_copy:
        paragraphs[paragraph["order"]] = paragraph["text"]
    del paragraphs_copy
    tokens = []
    tags = []
    
    paragraph_mapping = False
    paragraph_mapping_orders = []
    if "paragraph" in list(mapping.keys()) and len(mapping["paragraph"].keys()) != 0:
        paragraph_mapping = True
        paragraph_mapping_orders = list(mapping["paragraph"].keys())

    order_list = list(map(int, paragraphs.keys()))
    # order_list.pop()
    # print(order_list)
    for order in order_list:
        text = paragraphs[order]
        prev_is_whitespace = True
        answer_indexs = None
        if paragraph_mapping and str(order) in paragraph_mapping_orders:
            answer_indexs = mapping["paragraph"][str(order)]
        current_tags = [0 for i in range(len(text))]
        if answer_indexs is not None:
            for answer_index in answer_indexs:
                # mapping_content.append(text[answer_index[0]:answer_index[1]])
                current_tags[answer_index[0]:answer_index[1]] = \
                    [1 for i in range(len(current_tags[answer_index[0]:answer_index[1]]))]
        start_index = 0
        wait_add = False
        # change tag 1 to 2 and apply BIO tagging

        for i, c in enumerate(text):
            # print("c:", c, "wait_add:", wait_add,  "prev_is_whitespace:", prev_is_whitespace   , 'Index:', start_index)
            if is_whitespace(c):  # or c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                prev_is_whitespace = True
                
            elif c in ["-", "–", "~"]:
                if wait_add:
                    if 1 in current_tags[start_index:i]:
                        tags.append(1)
                    else:
                        tags.append(0)
                    wait_add = False
                tokens.append(c)
                tags.append(0)
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    tokens.append(c)
                    wait_add = True
                    start_index = i
                else:
                    tokens[-1] += c
                prev_is_whitespace = False
        if wait_add:
            if 1 in current_tags[start_index:len(text)]:
                tags.append(1)
            else:
                tags.append(0)

    # print(text)
    # print(tokens)
    # print(tags)

    # token = []
    # for i in tokens:
    #     num = to_number(i)
    #     if num is not None:
    #         token.append(str(num))
    #     else:        
    #         token.append(i)   

    # new_tags = []
    # zero = True
    # for idx, i in enumerate(tags):
    #     if i == 2 and zero == True:
    #         tags[idx] = 1
    #         # new_tags.append(1)
    #         zero = False
    #     elif i == 0:
    #         zero = True
    
    return tokens, tags


count = 0
for i in tqdm(range(len(data))):
    paragraphs = data[i]['paragraphs']
    uid = data[i]['table']['uid']
    text = [para['text'] for para in paragraphs]

    # if uid !=  '13bb283b-4b9c-42b9-9b02-f1b2e1a87abf': continue 
    # if uid !=  '32edf644-acb0-4260-9392-f0baa4253f5a': continue 
    # if uid !=  '3ffd9053-a45d-491c-957a-1b2fa0af0570': continue 

    questions = data[i]['questions']
    for ques in questions:
        ques_order = ques['order']
        mappings = ques['mapping']
        question = ques['question']

        # if ques_order != 3: continue
        # if ques_order != 6: continue

        if ques["answer_from"] == "table":
            continue
        
    
        para_tokens, para_tags = paragraph_tags(paragraphs, mappings)
        ques_token, ques_tag = question_tags(question)     

        if len(para_tokens) != len(para_tags) or len(ques_token) != len(ques_tag):
            f.write('Continuing\n')
            f.write(f"UID: {uid}\n")
            f.write(f"ORDER: {ques_order}\n")
            f.write(f"QUESTION: {question}\n")
            f.write(f"MAPPING: {mappings}\n")
            f.write('=' * 60 + '\n')
            continue
        

        # bert
        # tokens = ques_token + ['[SEP]'] + para_tokens
        # roberta
        tokens = ques_token + ['</s>'] + para_tokens
        
        tags = ques_tag + [None] + para_tags 

        row = {
                'uid':uid,
                'order': ques_order,
                'question':question,
                'text':' '.join(text),
                'answer': ques['answer'], 
                'token': tokens, 
                'ner': tags
        }

        dataset = dataset.append(row, ignore_index = True)

        # tags.remove(None)
        # if sum(tags) == 0:
        #     count += 1

print(count)
dataset.to_csv(f'dataset_tagop/{mode}_{suffix}.csv', index = False)
f.close()





# para_tokens, para_tags = paragraph_tags(paragraphs, mappings)
# type: ignore

import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ['WANDB_PROJECT']='TATQA-TEXT'

import json
import ast
import numpy as np
import pandas as pd 
import evaluate
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



batch_size = 32
nepochs = 15
max_length = 512
truncation = True
training_name = "bart-large-bs-32-only-text-ml-512-epoch-15"
model_checkpoint = "facebook/bart-large"

suffix = '_text'

print(model_checkpoint)
print(training_name)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)



metric = evaluate.load("squad")
# metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    gt = []
    pred = []
    
    for i in range(len(preds)):

        gt_dict = {'id' : str(i)}
        gt_dict['answers'] = {"text": [labels[i].strip()], 'answer_start':[0]}
        gt.append(gt_dict)

        pred_dict = {'id' : str(i)}
        pred_dict['prediction_text'] = preds[i].strip()
        pred.append(pred_dict)
        
    return pred, gt

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    # result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result   


class TextDataset(Dataset):
    def __init__(self, mode = 'train'):
        self.mode = mode
        if len(suffix):
            self.df = pd.read_csv(f'dataset_tagop/{mode}{suffix}.csv')
        else:
            self.df = pd.read_csv(f'dataset_tagop/{mode}.csv')
        self.tokenizer = tokenizer
        
        

    def __getitem__(self, idx):
        item = self.df.iloc[idx].values
        
        targets = item[-1]
        inputs = item[-2]
        
        tokenized_inputs = tokenizer(inputs, text_target = targets, max_length = max_length, truncation = truncation)
        return tokenized_inputs


    def __len__(self):
        return len(self.df)


train_dataset = TextDataset(mode = 'train')
eval_dataset = TextDataset(mode = 'dev')


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)



args = Seq2SeqTrainingArguments(
    training_name,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=nepochs,
    predict_with_generate=True,
    fp16=True,

    
    # evaluation_strategy = 'epoch',
    # save_strategy = 'epoch',

    # metric_for_best_model = "eval_loss",
    # greater_is_better = False, 

    # save_total_limit = 1,
    # load_best_model_at_end=True,

    evaluation_strategy = "epoch",
    save_total_limit = 2,
    save_strategy = "no",
    load_best_model_at_end=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)


train_result = trainer.train()
trainer.save_model(training_name)

# compute train results
metrics = train_result.metrics
metrics["train_samples"] = len(train_dataset)

# save train results
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)

# compute evaluation results
metrics = trainer.evaluate()
metrics["eval_samples"] = len(eval_dataset)

# save evaluation results
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

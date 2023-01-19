import os
os.environ["CUDA_VISIBLE_DEVICES"]="6"
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
nepochs = 10
max_length = 512
truncation = True
training_name = "t5-base-bs-32-only-text-count"
model_checkpoint = "t5-base"

suffix = '_count_text'

print(model_checkpoint)
print(training_name)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

 
# fmetric = evaluate.load("f1")
# pmetric = evaluate.load("precision")
# rmetric = evaluate.load("recall")
# ametric = evaluate.load("accuracy")

# def compute_metrics(eval_preds):
#     logits, labels = eval_preds
#     predictions = np.argmax(logits, axis=-1)
#     # print(predictions)
#     # print(labels)
#     a = ametric.compute(predictions=predictions, references=labels)
#     f = fmetric.compute(predictions=predictions, references=labels, average = 'macro')
#     p = pmetric.compute(predictions=predictions, references=labels, average = 'macro')
#     r = rmetric.compute(predictions=predictions, references=labels, average = 'macro')
#     return {**a, **f, **p, **r}


metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

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
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result   


class TextDataset(Dataset):
    def __init__(self, mode = 'train'):
        self.mode = mode
        if len(suffix):
            # print(f'dataset_tagop/{mode}{suffix}.csv')
            self.df = pd.read_csv(f'dataset_tagop/{mode}{suffix}.csv')
        else:
            self.df = pd.read_csv(f'dataset_tagop/{mode}.csv')
        self.tokenizer = tokenizer
        # self.prefix = "summarize: "
        # self.prefix = "translate English to English: "
            
        

    def __getitem__(self, idx):
        item = self.df.iloc[idx].values
        
        # question = item[-3]
        # text = item[-2]
        targets = item[-1]
        inputs = item[-2]
        # inputs = f"question: {question} context: {text}"
        
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
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=nepochs,
    predict_with_generate=True,
    fp16=True,

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

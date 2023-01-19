# type: ignore
import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ['WANDB_PROJECT']='TATQA-TEXT'


import json
import ast 
import evaluate
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoConfig

batch_size = 42
num_classes = 5
training_name = "roberta-bs-42-scale"
# model_checkpoint = "xlnet-base-cased"
model_checkpoint = "roberta-base"
# model_checkpoint = "distilbert-base-uncased"
# model_checkpoint = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
 
fmetric = evaluate.load("f1")
pmetric = evaluate.load("precision")
rmetric = evaluate.load("recall")
ametric = evaluate.load("accuracy") 

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    a = ametric.compute(predictions=predictions, references=labels)
    f = fmetric.compute(predictions=predictions, references=labels, average = 'macro')
    p = pmetric.compute(predictions=predictions, references=labels, average = 'macro')
    r = rmetric.compute(predictions=predictions, references=labels, average = 'macro')
    return {**a, **f, **p, **r}

class TextDataset(Dataset):
    def __init__(self, mode = 'train'):
        self.mode = mode
        self.df = pd.read_csv(f'dataset_tagop/{mode}_roberta.csv')
        # self.df = pd.read_csv(f'dataset_tagop/{mode}_class.csv')
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx].values
        tokenized_inputs = tokenizer(item[-2], truncation=True)
        tokenized_inputs['label'] = int(item[-1])
        return tokenized_inputs


    def __len__(self):
        return len(self.df)


train_dataset = TextDataset(mode = 'train')
eval_dataset = TextDataset(mode = 'dev')


# label2id = {"SPAN-TEXT": 0, "SPAN-TABLE": 1, "MULTI_SPAN": 2, "CHANGE_RATIO": 3,
#                     "AVERAGE": 4, "COUNT": 5, "SUM": 6, "DIFF": 7, "TIMES": 8, "DIVIDE": 9}

# id2label = {str(value): key for key,value in label2id.items()}
# config = AutoConfig.from_pretrained(model_checkpoint, label2id = label2id, id2label = id2label)
# print(config)
# model = AutoModelForSequenceClassification.from_config(config)


model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_classes)

training_args = TrainingArguments(
    training_name,
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    # learning_rate=0.0001,
    learning_rate=2e-5,
    num_train_epochs=20,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    weight_decay=0.01,
    # auto_find_batch_size = True


    save_total_limit = 2,
    save_strategy = "no",
    load_best_model_at_end=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
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
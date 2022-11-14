import os
os.environ['WANDB_DISABLED'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]="6"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
# os.environ['WANDB_PROJECT']='TATQA-TEXT'

import json
import ast 
import evaluate
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForTokenClassification, AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer



batch_size = 16
epochs = 20
training_name = 'bert-large-num-bs-16'
model_checkpoint = "bert-large-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

df = pd.read_csv('dataset_tagop/train.csv').values
# label_names = ['O', 'I']
label_names = ['O', 'I-ANS', 'B-ANS']

metric = evaluate.load("seqeval")


def align_labels_with_tokens(labels, word_ids, sep_idx):

    new_labels = []
    current_word = None
    for idx, word_id in enumerate(word_ids):
        if idx == sep_idx:
            new_labels.append(-100)
        elif word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


class TextDataset(Dataset):
    def __init__(self, mode = 'train'):
        self.mode = mode
        self.df = pd.read_csv(f'dataset_tagop/{mode}.csv')
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = self.df.iloc[idx].values

        tokenized_inputs = tokenizer(ast.literal_eval(item[-2]), truncation=True, is_split_into_words=True)
        labels =  ast.literal_eval(item[-1])            
        sep_idx = tokenized_inputs.tokens().index('[SEP]')
        word_ids = tokenized_inputs.word_ids()
        new_labels = align_labels_with_tokens(labels, word_ids, sep_idx)
        tokenized_inputs["labels"] = new_labels
        # print(tokenized_inputs.keys())
        # print('token', len(ast.literal_eval(item[-2])))
        # print('label', len(ast.literal_eval(item[-1])))
        # for key,val in tokenized_inputs.items():
        #     print(f'==== {key} ====')
        #     print(val, len(val))
        #     print('='*40)
        return tokenized_inputs


    def __len__(self):
        return len(self.df)


train_dataset = TextDataset(mode = 'train')
eval_dataset = TextDataset(mode = 'dev')
# train_dataset[0]
# exit(0cl)


id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}


model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    num_labels = len(label_names)
)



print(model.config.num_labels)

args = TrainingArguments(
    training_name,
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=epochs,
    weight_decay=0.01,
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size = batch_size,
    # auto_find_batch_size = True,
    # logging_dir = 'logs' 

    save_total_limit = 2,
    save_strategy = "no",
    load_best_model_at_end=False,
)


trainer = Trainer(
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

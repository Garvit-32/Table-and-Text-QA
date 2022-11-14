
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import ast
import torch 
import evaluate
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import TapasForQuestionAnswering, AdamW, TapasConfig, TapasTokenizer
from transformers import TrainingArguments, Trainer, AutoConfig
from transformers import DataCollatorWithPadding
# import wandb

batch_size = 32
n_epochs = 20
lr = 5e-5
dataset_path = 'dataset_tagop_table'
saved_path = 'tapas_large'
training_name = saved_path
os.makedirs(saved_path, exist_ok=True)

print('= '*60)
print(dataset_path)
print(saved_path)
print('= '*60)

# wandb.init(
#     project=f"TATQA-TAPAS",
#     name='test-run',
# )   


class TableDataset(Dataset):
    def __init__(self, tokenizer, mode = 'train'):
        self.mode = mode
        self.df = pd.read_csv(f'{dataset_path}/{mode}.csv')
        self.tokenizer = tokenizer
        self.table_csv_path = f'{dataset_path}/{mode}'

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        table = pd.read_csv(f'{self.table_csv_path}/{item.uid}.csv').astype(str) # TapasTokenizer expects the table data to be text only

        encoding = self.tokenizer(table=table, 
                                queries=item.question, 
                                answer_coordinates=ast.literal_eval(item.answer_cord), 
                                answer_text=ast.literal_eval(item.answer),
                                # padding="max_length",
                                truncation=True,
                                return_tensors="pt"
                                
        )

        # remove the batch dimension which the tokenizer adds 
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
    
        encoding['labels'] = encoding['labels'].float()
        encoding['uid'] = item.uid + '+' + str(item.position)
        
        return encoding

    def __len__(self):
        return len(self.df)
    

model_name = "google/tapas-large-finetuned-sqa"
tokenizer = TapasTokenizer.from_pretrained(model_name)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
metric = evaluate.load("squad_v2")

train_dataset = TableDataset(tokenizer=tokenizer, mode = 'train')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle = False)

# train_dataset[0]
eval_dataset = TableDataset(tokenizer=tokenizer, mode = 'dev')
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle = False)

df = pd.read_csv(f'{dataset_path}/dev.csv')

# config = TapasConfig(select_one_column = False)
model = TapasForQuestionAnswering.from_pretrained(model_name, select_one_column = False, allow_empty_column_selection=True)
# mimic from squadv2 dataset

training_args = TrainingArguments(
    training_name,
    evaluation_strategy="epoch",
    # save_strategy="epoch",
    # learning_rate=0.0001,
    learning_rate=lr,
    num_train_epochs=n_epochs,
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


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# optimizer = AdamW(model.parameters(), lr=lr)

# best_loss = float('inf')

def get_answers(uid, pred_answer_coord):

    pred_answers = []
    gt_answers = []

    for idx, coordinates in enumerate(pred_answer_coord[0]):

        table_uid = uid[idx].split('+')[0]
        question_position = int(uid[idx].split('+')[1])
        table = pd.read_csv(f'{dataset_path}/dev/{table_uid}.csv').astype(str)
        # print(table)

        ##############################################################

        gt_dict = {'id' : uid[idx]}

        gt_answers_coords = ast.literal_eval(df[(df.uid == table_uid) & (df.position == question_position)].answer_cord.values[0])

        # print('='*60)
        # print('gt', gt_answers_coords)
        # print('pred', pred_answer_coord)
        # print('='*60)

        if len(gt_answers_coords) == 0: 
            # continue
            gt_dict['answers'] = {"text": [''], 'answer_start':[]}

            # gt_dict['answers'] = {"text": [table.iat[gt_answers_coords[0]]], 'answer_start':[0]}
        

        else:
            cell_values = []
            for coordinate in gt_answers_coords:
                cell_values.append(table.iat[coordinate])
            gt_dict['answers'] = {"text": [", ".join(cell_values)], 'answer_start':[0]}


        gt_answers.append(gt_dict)
        ##############################################################
        # print('pred', coordinates)
        pred_dict = {'id' : uid[idx]}
        if len(coordinates) == 0: 
            pred_dict['prediction_text'] = ''
            pred_dict['no_answer_probability'] = 1.
            
        # only a single cell:
        if len(coordinates) == 1:
            pred_dict['prediction_text'] = table.iat[coordinates[0]]
            pred_dict['no_answer_probability'] = 0.

        # multiple cells
        else:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            pred_dict['prediction_text'] = ", ".join(cell_values)
            pred_dict['no_answer_probability'] = 0.

        pred_answers.append(pred_dict)  
        ##############################################################


    # print(pred_answers)
    # print(gt_answers)
    return pred_answers, gt_answers






# for epoch in range(1, n_epochs + 1):


#     test_loss = 0.0
#     train_loss = 0.0

#     test_exact = 0.0
#     test_f1 = 0.0
    
#     model.train()
#     for idx, batch in tqdm(enumerate(train_dataloader), total = len(train_dataloader)):
#         # get the inputs;
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         token_type_ids = batch["token_type_ids"].to(device)
#         labels = batch["labels"].to(device)
#         # zero the parameter gradients
#         optimizer.zero_grad()
#         # forward + backward + optimize
#         outputs = model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             labels=labels,
#         )
#         loss = outputs.loss

#         train_loss += loss.item()
#         # print(idx, loss.item())

#         loss.backward()
#         optimizer.step()

#         # wandb.log({'Train Batch Loss': loss.item()})

#     with torch.no_grad():
#         for idx, batch in enumerate(eval_dataloader):

#             input_ids = batch["input_ids"].to(device)
#             attention_mask = batch["attention_mask"].to(device)
#             token_type_ids = batch["token_type_ids"].to(device)
#             labels = batch["labels"].float().to(device)
#             uid = batch.pop('uid')

#             outputs = model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids,
#                 labels=labels,
#             )

#             loss = outputs.loss

#             predicted_answer_coordinates = tokenizer.convert_logits_to_predictions(batch, outputs.logits.detach().cpu())
#             predicted_answers, gt_answers = get_answers(uid, predicted_answer_coordinates)
            
#             result = metric.compute(predictions=predicted_answers, references=gt_answers)

#             test_loss += loss.item()
#             test_exact += result['exact_match']
#             test_f1 += result['f1']
#             # wandb.log({'Valid Batch Loss': loss.item()})

#             print(f'[{epoch}/{n_epochs}] \t [{idx}/{len(eval_dataloader)}] \t Loss: {loss.item():.4f}')


#     if best_loss > test_loss/len(eval_dataloader):
#         best_loss = test_loss/len(eval_dataloader)
#         model.save_pretrained(f"{saved_path}/best_save_{epoch}")

#     print(f'Epoch: {epoch}/{n_epochs} \t Training Loss: {train_loss/len(train_dataloader):.4f} \t Testing Loss: {test_loss/len(eval_dataloader):.4f}')
#     print(f'Epoch: {epoch}/{n_epochs} \t Exact Match: {test_exact/len(eval_dataloader):.4f} \t F1 Score: {test_f1/len(eval_dataloader):.4f}')

#     # wandb.log({
#     #     'Train Epoch Loss': train_loss/len(train_dataloader), 
#     #     'Valid Epoch Loss': test_loss/len(eval_dataloader), 
#     #     'F1 Score': test_exact/len(eval_dataloader),
#     #     'Exact Match': test_f1/len(eval_dataloader),
#     # })
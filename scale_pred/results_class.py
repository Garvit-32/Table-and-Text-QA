import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# target_names = ["TEXT", "CHANGE_RATIO", "AVERAGE", "COUNT", "SUM", "DIFF", "TIMES", "DIVIDE" ,"MIXED"]

target_names = ["", "thousand", "million",  "billion", "percent"]

    # bert-finetuned-ner
# model_checkpoint = "roberta-bs-42-scale"
model_checkpoint = "bert-tc-bs-42-scale"
text_classifier = pipeline(
    "text-classification", model=model_checkpoint, device = 0, truncation= True
)

# 4232c6c1-97cf-48ad-8b8b-f956871a3212,4

# print(text_classifier('What is the difference between average salaries and fees and average incentive schemes from 2018 to 2019?'))

df = pd.read_csv("dataset_tagop/dev_class.csv").values

y_true = []
y_pred = []

for i in tqdm(df):
    y_true.append(int(i[-1]))
    # print(text_classifier(i[-2])[0]['label'].split('_')[-1])
    pred = int(text_classifier(i[-2])[0]['label'].split('_')[-1])
    y_pred.append(pred)


print('  ') 
print(classification_report(y_true, y_pred, target_names = target_names))
print('Accuracy:', accuracy_score(y_true, y_pred))
print('F1 SCore:', f1_score(y_true, y_pred, average= 'macro'))

cm = confusion_matrix(y_true, y_pred)

fig = plt.figure(figsize = (10,10))
ax = sns.heatmap(cm, annot=True, fmt='g')

ax.set_title('Confusion Matrix')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

# ## For the Tick Labels, the labels should be in Alphabetical order
ax.xaxis.set_ticklabels(target_names, rotation = 45, ha="right")
ax.yaxis.set_ticklabels(target_names, rotation = 45, ha="right")


fig.savefig('cm.png')


# plt.show()
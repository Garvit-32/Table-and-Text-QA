import pandas as pd



df = pd.read_csv('dataset_tagop/dev_old.csv')
import numpy as np
x = []
for i in df.values:
    x.append(i[-1])

print(np.unique(x))
# print(df[df.label == 10].shape)


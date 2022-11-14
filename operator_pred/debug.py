import pandas as pd



df = pd.read_csv('dataset_tagop/dev.csv')


print(df[df.label == 10].shape)


import pandas as pd
import numpy as np
from tqdm import tqdm
import seaborn as sns
from transformers import pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

model_checkpoint = "t5-base-translation-bs-8/checkpoint-6500"
translator = pipeline(
    "translation_en_to_en", model=model_checkpoint, device = 0, max_length = 128
)

# print(translator('What is the percentage change in Other in 2019 from 2018? </s> 44.1 56.7 </s> - / ( ) [ ]'))
print(translator('For Balance payable as at June 30, 2019, What is the difference between Workforce reduction and Facility costs? </s> 1,046 2,949 </s> -'))
print(translator('For Balance payable as at June 30, 2019, What is the difference between Workforce reduction and Facility costs? </s> 1,0213 2,949 </s> -'))
print(translator('For Balance payable as at June 30, 2019, What is the difference between Workforce reduction and Facility costs? </s> 2,949 1,0213 </s> -'))


# en_fr_translator = pipeline("translation_en_to_fr")
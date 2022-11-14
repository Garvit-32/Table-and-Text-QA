from evaluate import load


# squad_metric = load("squad")
squad_metric = load("squad_v2")


# predictions = [{'prediction_text': '', 'id': '56e10a3be3433e1400422b22',},]
predictions = [{'prediction_text': '', 'id': '56e10a3be3433e1400422b22', 'no_answer_probability': 0.},]



references = [{'answers': {'answer_start': [], 'text': ['']}, 'id': '56e10a3be3433e1400422b22'}]
# references = [{'answers': {'answer_start': [], 'text': ['']}, 'id': '56e10a3be3433e1400422b22'}, {'answers': {'answer_start': [233], 'text': ['Beyoncé and Bruno Mars']}, 'id': '56d2051ce7d4791d0090260b'}, {'answers': {'answer_start': [891], 'text': ['climate change']}, 'id': '5733b5344776f419006610e1'}]

# results = squad_metric.compute(predictions=predictions, references=references)
# print(results)



import numpy as np

print(np.mean([1,2]))



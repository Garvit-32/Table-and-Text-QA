import pandas as pd
import ast

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
        # print('gt', gt_answers_coords)
        if len(gt_answers_coords) == 0: 
            gt_dict['answers'] = {"text": [table.iat[gt_answers_coords[0]]], 'answer_start':[0]}

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
            
        # only a single cell:
        if len(coordinates) == 1:
            pred_dict['prediction_text'] = table.iat[coordinates[0]]
        # multiple cells
        else:
            cell_values = []
            for coordinate in coordinates:
                cell_values.append(table.iat[coordinate])
            pred_dict['prediction_text'] = ", ".join(cell_values)

        pred_answers.append(pred_dict)  
        ##############################################################


    # print(pred_answers)
    # print(gt_answers)
    return pred_answers, gt_answers
#!/usr/bin/env python
# coding: utf-8
import os
import openai
from time import time, sleep
import numpy as np
import pickle as pkl
import json
import matplotlib.pyplot as plt
import re
from copy import deepcopy as copy
import sys
sys.path.insert(0, './')
from src import dataset, gpt
from src.multiprocess import multiprocess
from src.str2arithmatic import run_expression, get_last_expression, get_last_expression_insert, replace_with_calculator_results

gpt_model='gpt-3.5-0301'
#Need to change to 'gpt-3.5-0613'
gpt=gpt.GPT(gpt_model)

from ipywidgets import IntProgress
from IPython.display import display
import argparse

#Load pkl file
class args:
    dataset='mathqa'
    output_folder='./output/mathqa/'
    generation_repeat_time=10
    start_id=0
    end_id=2985
    t=1.0
    ablation_num=100
    output_folder_ablation='./output_ablation/'
    
file_names=[name for name in os.listdir(args.output_folder) if str(args.generation_repeat_time)+'_finish_step3.4' in name]

id_triple_list=[]
for i in range(len(file_names)):
    file_name=file_names[i].split('_')
    id_triple_list.append([i, int(file_name[0]), int(file_name[1])])
id_triple_list=sorted(id_triple_list, key=lambda x: x[1])

output=[]
current_need_start_id=args.start_id
for i in range(len(id_triple_list)):
    id, file_start_id, file_end_id=id_triple_list[i]
    if file_start_id>current_need_start_id:
        print('missing data from number: ', current_need_start_id)
        raise Exception("Failed")
    if file_end_id<current_need_start_id:
        continue
    file_output=pkl.load(open(args.output_folder+file_names[id], 'rb'))
    if file_end_id<args.end_id:
        output.extend(file_output[(current_need_start_id-file_start_id)*args.generation_repeat_time:])
        current_need_start_id=file_end_id
    else:
        output.extend(file_output[(current_need_start_id-file_start_id)*args.generation_repeat_time:(args.end_id-file_start_id)*args.generation_repeat_time])
        break

record_list_1=output

data=dataset.load_dataset('MathQA')['test']

if True:
    id_list=np.arange(len(data))
    np.random.seed(235)
    np.random.shuffle(id_list)
    data_new=[]
    for i in id_list:
        data_new.append(data[i])
    data=data_new


record_list=[]
for i in range(args.ablation_num):
    id_1=id_list[i]
    record_list.extend(record_list_1[id_1*10: (id_1+1)*10])

print(len(record_list))

# Finish data loading! Start generation.
def run_gpt(input, stop, max_tokens, temp):
    return gpt(input, stop=stop, max_tokens=max_tokens, temp=temp)

#['question', 'formalized_question', 'conditions', 'correct', 'steps', 'step_record', 'predicted_result', 'unknown_variable']
#print(record_list[0]['step_record'][0].keys()) ['status', 'original_step', 'target', 'condition_ids', 'step_ids', 'collection_raw', 'rerunning_step', 'comparison_result', 'verification_result_pre', 'verification_result']

prompt='''Here is some information:
“1. Alice gets 2 apples.
2. Alice gets twice as many oranges as bananas.
3. Original price per apple: $2
Discount: 50%
Price per apple after discount: $2 * (1 - 0.50) = $2 * 0.50 = $1"
Based on the given information, here is a reasoning process:
"Calculate Alice’s cost of the 5 apples after a 50% discount.
Price per apple after discount: $1
Apples be eaten: 3
Oranges be eaten: 6
Total apples: 5
Total oranges: x
Total bananas: 2x
Alice’s total number of fruits: 5 + x + 2x = 5 + 4x
Alice’s total cost of 5 apples: $1 * (5 - 3) = $2”

Double-check the reasoning process, let’s analyze its correctness, and end with "yes" or "no".
Answer:
Let’s think step by step.
Grounding check
Purpose: Find references for numbers in the reasoning process sequentially
Rule: Any ungrounded number makes the check fails
"x" is a variable, we do not care about variables.
"y" is a variable, we do not care about variables.
"5" is a quantity from the given information that represents the number of apples.
"50%" is a quantity from the given information that represents the discount.
"$1" is a quantity from the given information that represents the price per apple after the discount.
"3" is a quantity not from the given information, it is ungrounded.
"6" is a quantity not from the given information, it is ungrounded.
"$2" is a quantity from the given information that represents the original price per apple.
Result: Based on the check, there are ungrounded numbers, so the grounding check fails.
Reasoning check:
Purpose: Check whether the reasoning logic correctly answers the question
Rule: Any logic error makes the check fails
To calculate the total cost of apples after a 50% discount, we should multiply the number of apples
by the discounted price. But to answer the question, we do not need to calculate the total number
of fruit Alice gets.
Result: Based on the check, there are some logic mistakes, so the reasoning check fails.
Calculation check:
Purpose: Check whether the calculation process is consistent
Rule: Any inconsistent calculation makes the check fails
calculation1:
equation: $1 * (5 - 3), answer: $2
(5 - 3) = 2
$1 * 2 = $2 is consistent with the answer, so the calculation is correct.
calculation2:
equation: 5 + x + 2x, answer: 5 + 4x
x + 2x = 3x
5 + 3x is inconsistent with the answer, so the calculation is incorrect.
Result: Based on the check, the calculation process is inconsistent, so the calculation check fails.
Check results: Ground check fails, Reasoning check fails, Calculation check fails.
Rule: Any failed check makes the reasoning incorrect.
So the answer is "no".'''


parameter_list=[]
working_id_list=[]
for i in range(len(record_list)):
    for j in range(len(record_list[i]['step_record'])):
        input=prompt+'\n\n'
        input+='Here is some information:\n"'
        k=1
        for id in record_list[i]['step_record'][j]['condition_ids']:
            if id<len(record_list[i]['conditions']):
                input+=record_list[i]['conditions'][id]
                k+=1
        for id in record_list[i]['step_record'][j]['step_ids']:
            if id < len(record_list[i]['steps']):
                input+=str(k)+'.'+record_list[i]['steps'][id]+'\n'
                k+=1
        input+='"\nBased on the given information, here is a reasoning process:\n"'
        input+=record_list[i]['steps'][j]+'"'
        input+='\n\nDouble-check the reasoning process, let’s analyze its correctness, and end with "yes" or "no".\n\nAnswer:'
        parameter_list.append([input, None, 1000, args.t])
        working_id_list.append([i, j])

output=multiprocess(run_gpt, 10, parameter_list, 'tqdm')

for ii in range(len(output)):
    i, j = working_id_list[ii]
    record_list[i]['step_record'][j]['verification_result_pre']=output[ii]
    
    conclusion=output[ii].split('\n')[-1].lower()
    if 'no' in conclusion:
        record_list[i]['step_record'][j]['verification_result']=-1
    elif 'yes' in conclusion:
        record_list[i]['step_record'][j]['verification_result']=1
    else:
        record_list[i]['step_record'][j]['verification_result']=0

pkl.dump(record_list, open(args.output_folder_ablation+'error_check_one_shot.pkl', 'wb'))


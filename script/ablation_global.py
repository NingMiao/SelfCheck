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

prompt='The following is a question and a solution to it from a student. Carefully check whether the solution is correct step by step. End your response with your conclusion that starts with "Correct", "Wrong" or "Not Sure".'
parameter_list=[]
for i in range(len(record_list)):
    question=record_list[i]['question']
    input=prompt+'\n\nQuestion: '+question+'\n\nSolution: '+'\n'.join(record_list[i]['steps'])
    parameter_list.append([input, None, 1000, args.t])

output=multiprocess(run_gpt, 10, parameter_list, 'tqdm')
for i in range(len(record_list)):
    record_list[i]['raw_global_verification_result']=output[i]
    
    conclusion=output[i].split('\n')[-1].lower()
    if 'wrong' in conclusion:
        record_list[i]['global_verification_result']=-1
    elif 'correct' in conclusion:
        record_list[i]['global_verification_result']=1
    else:
        record_list[i]['global_verification_result']=0

pkl.dump(record_list, open(args.output_folder_ablation+'global.pkl', 'wb'))


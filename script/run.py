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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['gsm8k', 'math', 'mathqa', 'prm800k'], required=True)
    parser.add_argument("--start_id", type=int, default=0, help="Start id", required=False)
    parser.add_argument("--end_id", type=int, default=0, help="End id (not included)", required=False)
    parser.add_argument("--t", type=float, default=1.0, help="temperature", required=False)
    parser.add_argument("--generation_repeat_time", type=int, default=10, required=False)
    parser.add_argument("--output_folder", type=str, default='./output/', required=False)
    parser.add_argument("--starting_from", type=float, default=1.0, choices=[1.0, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2, 3.3, 3.4], required=False)
    parser.add_argument("--data_select_random_seed", type=int, default=0, required=False)
    # parse
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    args.output_folder=args.output_folder+args.dataset+'/'
    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    return args


args=parse_args()


#Loading data

if args.dataset=='gsm8k':
    data=dataset.load_dataset('GSM8K')['test']
elif args.dataset=='mathqa':
    data=dataset.load_dataset('MathQA')['test']
elif args.dataset=='math':
    data=dataset.load_dataset('Math')['test']
elif args.dataset=='prm800k':
    data=dataset.load_dataset('prm800k', phase=1)['test']

if args.data_select_random_seed>0:
    id_list=np.arange(len(data))
    #!The current seeds is 235.
    np.random.seed(args.data_select_random_seed)
    np.random.shuffle(id_list)
    data_new=[]
    for i in id_list:
        data_new.append(data[i])
    data=data_new
    
    
data_len=len(data)
if args.end_id==0 or args.end_id>=data_len:
    args.end_id=len(data)
data=data[args.start_id: args.end_id]
    

# Finish data loading! Start generation.
def run_gpt(input, stop, max_tokens, temp):
    return gpt(input, stop=stop, max_tokens=max_tokens, temp=temp)


#Step 1.0: Generate multiple result for chosing
if args.dataset!='prm800k' and args.starting_from<=1.0:
    if 'options' in data[0]:
        prompt='Solve the following problem step by step. Please start each step with "Step :" and split sentences in each step with "\n\n". Please choose one from options "a )", "b )", "c )", "d )" and "e )" at the end of your reply.'
        parameter_list=[]
        for i in range(len(data)):
            question=data[i]['question']
            options=data[i]['options']
            input=prompt+'\nProblem: '+question
            input+='\n\n'
            input+='Options: '+options
            parameter_list.append([input, None, 1000, args.t])
    
    else:
        prompt='Solve the following problem step by step. Please start each step with "Step :" and split sentences in each step with "\n\n". Please finish you response with "So the answer is ...".'
        parameter_list=[]
        for i in range(len(data)):
            question=data[i]['question']
            input=prompt+'\nProblem: '+question
            parameter_list.append([input, None, 1000, args.t])


    candidate_list=[]
    for repeat_id in range(args.generation_repeat_time):
        output=multiprocess(run_gpt, 10, parameter_list, 'tqdm')
        candidate_list.append(output)

    pkl.dump(candidate_list, open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step1.0.pkl', 'wb'))
elif args.dataset=='prm800k':
    candidate_list=[0]
elif args.starting_from in [2.0, 2.1]:
    candidate_list=pkl.load( open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step1.0.pkl', 'rb'))

#Step 2.0: Get formalized question and split 
if args.starting_from<=2.0:
    prompt='Faithfully copy the following paragraph, and add "\s" at the beginning of each sentence.'
    parameter_list=[]
    for i in range(len(data)):
        question=data[i]['question']
        input=prompt+'\n\n'+question
        parameter_list.append([input, None, 500, args.t])

    formalized_question=multiprocess(run_gpt, 10, parameter_list, 'tqdm')
    pkl.dump(formalized_question, open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.0.pkl', 'wb'))
elif args.starting_from>=2.1:
    formalized_question=pkl.load( open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.0.pkl', 'rb'))


#Step 2.1 Initialize record_list
def remove_step(line):
    for i in range(len(line)):
        if line[i]==':':
            for j in range(i+1, len(line)):
                if line[j]!=' ':
                    return line[j:]
    return ''

def split_into_steps(reasoning):
    reasoning=reasoning.split('\n')
    steps=[]
    for line in reasoning:
        if len(line)==0:
            continue
        elif line.lower().startswith('step') or len(steps)==0:
            steps.append([remove_step(line)])
        elif len(steps)==0:
            steps.append([line])
        else:
            steps[-1].append(line)
    steps=['\n'.join(step) for step in steps]
    
    steps_=[]
    for step in steps:
        if len(steps_)==0:
            steps_.append(step)
        elif steps_[-1].strip().endswith(':'):
            steps_[-1]+=step
        else:
            steps_.append(step)
    return steps_

def get_conditions(question):
    conditions=question.split('\s')
    return conditions[1:]

if args.starting_from<=2.1:
    record_list=[]
    for i in range(len(data)):
        for j in range(len(candidate_list)):
            record={}
            record['question']=data[i]['question']
            if 'options' in data[i]:
                record['options']=data[i]['options']
            record['formalized_question']=formalized_question[i]
            record['conditions']=get_conditions(record['formalized_question'])
        
            if args.dataset!='prm800k':
                record['correct']=data[i]['answer']
                candidates=split_into_steps(candidate_list[j][i])
                record['steps']=candidates
            else:
                record['steps']=[item['main_step'] for item in data[i]['steps']]
                #record['human_verification_result']=data[i]['verification_result']
            record_list.append(record)

    #Initialize step_record in each record
    for i in range(len(record_list)):
        record_list[i]['step_record']=[]
        if args.dataset!='prm800k':
            for j in range(len(record_list[i]['steps'])):
                step_record={'status': 'initialized',
                     'original_step': record_list[i]['steps'][j]}
                record_list[i]['step_record'].append(step_record)
        else:
            for j in range(len(data[i]['steps'])):
                for k in range(len(data[i]['steps'][j]['candidates'])):
                    step_record={'status': 'initialized',
                         'original_step': data[i]['steps'][j]['candidates'][k],
                         'human_verification_result': data[i]['steps'][j]['verification_result'][k],
                         'step_id': j}
                    record_list[i]['step_record'].append(step_record)        
        

        
    if args.dataset=='math':
        from data.MATH.utils import postprocess_answer, extract_math_answer
        for record in record_list:
            try:
                record['predicted_result'] = postprocess_answer(extract_math_answer(record['steps'][-1]))
            except:
                try:
                    record['predicted_result'] = postprocess_answer(record['steps'][-1])
                except:
                    record['predicted_result'] = record['steps'][-1]
            record['correct']=postprocess_answer(record['correct'])
    elif args.dataset=='mathqa':
        from data.MathQA.utils import get_option_id
        for record in record_list:
            record['predicted_result']=get_option_id(record['steps'][-1])
    elif args.dataset=='gsm8k':
        from data.GSM8K.utils import get_answer
        for record in record_list:
            record['predicted_result']=get_answer(record['steps'][-1])
    pkl.dump(record_list, open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.1.pkl', 'wb'))
elif args.starting_from==2.2:
    record_list=pkl.load( open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.1.pkl', 'rb'))
    

#Step 2.2: Collecting unknown_variable
def input_for_unknown_variable(question, steps):
    string='The following is the solution to a math question: \n\n'
    
    string+='Question: '+question+'\n\n'
    string+='Solution: '+'\n\n'.join(steps)+'\n\n'
    string+='Are there variables in the solution? If so, please list the definition of variable in the form of "1. {variable} is defined as...". '

    return string

def read_unknown_variable(output):
    variable_def_list=[]
    for line in output.split('\n'):
        if len(line)>=1 and line[0].isdigit():
            variable_def_list.append(line)
    return variable_def_list

if args.starting_from<=2.2:
    parameter_list=[]
    for record in record_list:
        input=input_for_unknown_variable(record['question'], record['steps'])
        parameter_list.append([input, None, 1000, 0.0])
    output=multiprocess(run_gpt, 10, parameter_list, 'tqdm')

    for i in range(len(record_list)):
        record_list[i]['unknown_variable']=read_unknown_variable(output[i])

    pkl.dump(record_list, open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.2.pkl', 'wb'))
elif args.starting_from==3.0:
    record_list=pkl.load( open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step2.2.pkl', 'rb'))


# Step 3.0: Preparation finished! Start verification.

#Functions for verification
def input_for_target_extraction(question, steps, i):
    string='The following is a part of the solution to the problem: '
    string+=question
    string+='\n\n'
    string+='\n\n'.join(steps[:i+1])
    #string+='\n\nWhat is the aim of the last step "'
    string+='\n\nWhat specific action does the step "'
    string+=steps[i]
    string+='" take? Please give a brief answer using a single sentence and do not copy the steps.'
    return string   

def input_for_information_collection(question, steps, i, conditions):
    string='This is a math question: '
    string+='\n\nQuestion: '+question
    
    string+='\n\nThe following is information extracted from the question:\n\n'
    for j in range(len(conditions)):
        string+='Information '+str(j)+': '+conditions[j]+'\n'
    
    if i>=1:
        string+='\n'
        string+='The following is the first a few steps in a solution to the problem:'
        string+='\n\n'
        for j in range(i):
            string+='Step '+str(j)+': '+steps[j]+'\n'
    
        string+='\nWhich previous steps or information does the next step\n"'
        string+=steps[i]
        string+='"\ndirectly follow from?'
    else:
        string+='\nWhich information does the first reasoning step\n"'
        string+=steps[i]
        string+='"\ndirectly follow from?'
    return string  

def get_id_from_information_collection_output(output):
    output=output.lower()
    def get_id(string):
        def get_number(string):
            s=''
            for c in string:
                if c.isdigit():
                    s+=c
            return s
        
        #Deal with steps
        if len(string)>=2 and string[0]=='s':
            string=string[1:]
        elif len(string)<2:
            return []
        
        string=string.strip()
        id_list=[]
        
        word_list=string.split(' ')
        for word in word_list:
            number=get_number(word)
            if number != '':
                try:
                    id_list.append(int(number))
                except:
                    pass
            elif word in [',', 'and', 'or']:
                continue
            else:
                break

        return id_list
    
    information_id_list=[]
    for i in range(len(output)):
        if output[i:].startswith('information'):
            id_list=get_id(output[i+11:])
            if len(id_list)>=1:
                for id in id_list:
                    if id not in information_id_list:
                        information_id_list.append(id)
    
    step_id_list=[]
    for i in range(len(output)):
        if output[i:].startswith('step'):
            id_list=get_id(output[i+4:])
            if len(id_list)>=1:
                for id in id_list:
                    if id not in step_id_list:
                        step_id_list.append(id)
    
    return information_id_list, step_id_list 
            

def input_for_rerunning(target, conditions, steps, unknown_variable):
    string='We are in a process of solving a math problem.\n\n'
    
    if unknown_variable is not None:
        string+='Variables are defined as: \n'
        string+='\n'.join(unknown_variable)+'\n\n'
    
    if len(conditions)>=1:
        string+='We have some information from the problem: \n\n'
        for i in range(len(conditions)):
            string+='Information '+str(i)+': '+conditions[i]+'\n'
        string+='\n'
    
    if len(steps)>=1:
        string+='The following are some previous steps: \n\n'
        for i in range(len(steps)):
            string+='Previous step '+str(i)+': '+steps[i]+'\n'
        string+='\n\n'
    
    string+='The target for next step is: '
    string+='\n\n'
    string+=target
    string+='\n\n'
    
    if len(conditions)>=1 and len(steps)>=1:
        string+='Please try to achieve the target with the information from the problem or previous steps.'
    elif len(conditions)>=1 and len(steps)==0:
        string+='Please try to achieve the target with the information from the problem.'
    elif len(conditions)==0 and len(steps)>=0:
        string+='Please try to achieve the target with the information from previous steps.'
    else:
        string+='Please try to achieve the target.'
    return string

def input_for_compare(target, original_step, rerunning_step):
    string='The following are 2 solutions to a math problem:\n\n'
    string+='\n\n'
    string+='Solution 1: '
    string+=rerunning_step
    string+='\n\n'
    string+='Solution 2: '
    string+=original_step
    string+='\n\n'
    string+='Compare the key points from both solutions step by step and then check whether Solution 1 "supports", "contradicts" or "is not directly related to" the conclusion in Solution 2. Pay special attention to difference in numbers.'
    return string


def input_for_comparision_formalization(comparison_result):
    string='The following is the reply when we ask whether a solution "supports", "contradict" or "is not directly related to" another solution. Please use "Contradict", "Support" or "Not related" to summarize the final conclusion of the reply.'
    string+='\n\n'
    string+='Reply: '
    string+=comparison_result
    return string

def get_verification_result(verification_output, comparison_result):
    if 'contradict' in verification_output.lower():
        return -1
    elif 'support' in verification_output.lower():
        return 1
    else:
        return 0

if args.starting_from>=3.1:
    starting_step_=round(10*(args.starting_from-3.0))
    record_list = pkl.load( open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step3.'+str(starting_step_-1)+'.pkl', 'rb'))
    
else:
    starting_step_=0
    
for step_ in range(starting_step_, 10):
    #Pre-running
    parameter_list=[]
    id_record_list=[]
    for i in range(len(record_list)):
        record=record_list[i]
        for j in range(len(record['step_record'])):
            step_record=record['step_record'][j]
            if step_record['status']=='initialized':
                if record['question'] is None or  record['steps'] is None:
                    step_record['status']='finished'
                    step_record['verification_result']='error after initialized'
                else:
                    if args.dataset!='prm800k':
                        input=input_for_target_extraction(record['question'], record['steps'], j)
                    else:
                        input=input_for_target_extraction(record['question'], record['steps'], step_record['step_id'])
                    step_record['status']='target_extracting'
                    max_tokens=300
            elif step_record['status']=='target_extracted':
                if record['conditions'] is None:
                    step_record['status']='finished'
                    step_record['verification_result']='error after target_extracted'
                else:
                    if args.dataset!='prm800k':
                        input=input_for_information_collection(record['question'], record['steps'], j, record['conditions'])
                    else:
                        input=input_for_information_collection(record['question'], record['steps'], step_record['step_id'], record['conditions'])
                    step_record['status']='information_collecting'
                    max_tokens=300
            elif step_record['status']=='information_collected':
                related_conditions=[record['conditions'][k] for k in step_record['condition_ids'] if k<len(record['conditions'])]
                related_steps=[record['steps'][k] for k in step_record['step_ids'] if k<len(record['steps'])]
                input=input_for_rerunning(step_record['target'], related_conditions, related_steps, record['unknown_variable'])
                step_record['status']='rerunning'
                max_tokens=1000
            elif step_record['status']=='rerun':
                original_step=step_record['original_step']
                rerunning_step=step_record['rerunning_step']
                target=step_record['target']
                if rerunning_step is not None:
                    input=input_for_compare(target, original_step, rerunning_step)
                    step_record['status']='comparing'    
                    max_tokens=300
                else:
                    step_record['status']='finished'
                    step_record['verification_result']='error after rerun'
                    
                
            elif step_record['status']=='compared':
                comparison_result=step_record['comparison_result']
                if step_record['comparison_result'] is None:
                    step_record['status']='finished'
                    step_record['verification_result']='error after compared'
                else:
                    input=input_for_comparision_formalization(comparison_result)
                    step_record['status']='comparision_formalizing'
                    max_tokens=100
            if step_record['status']!='finished':
                parameter_list.append([input, None, max_tokens, args.t])
                id_record_list.append([i, j])
                
    #Running
    if len(parameter_list)==0:
        print('All finished!')
        break
    print('Start verification step: ', step_)
    output_list=multiprocess(run_gpt, 20, parameter_list, 'tqdm')

    #Post-running
    for id in range(len(id_record_list)):
        i, j=id_record_list[id]
        step_record=record_list[i]['step_record'][j]
        output=output_list[id]
        if step_record['status']=='target_extracting':
            step_record['target']=output
            step_record['status']='target_extracted'
        elif step_record['status']=='information_collecting':
            step_record['condition_ids'], step_record['step_ids']=get_id_from_information_collection_output(output)
            step_record['collection_raw']=output
            step_record['status']='information_collected'
        elif step_record['status']=='rerunning':
            step_record['rerunning_step']=output
            step_record['status']='rerun'
        elif step_record['status']=='comparing':
            step_record['comparison_result']=output
            step_record['status']='compared'
        elif step_record['status']=='comparision_formalizing':
            step_record['verification_result_pre']=output
            step_record['verification_result']=get_verification_result(output, step_record['comparison_result'])
            step_record['status']='finished'
    pkl.dump(record_list, open(args.output_folder+str(args.start_id)+'_'+str(args.end_id)+'_'+str(args.generation_repeat_time)+'_'+'finish_step3.'+str(step_)+'.pkl', 'wb'))
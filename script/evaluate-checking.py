#!/usr/bin/env python
# coding: utf-8
import os
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

import matplotlib.pyplot as plt

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['gsm8k', 'math', 'mathqa', 'prm800k', 'mathqa_train'], required=True)
    parser.add_argument("--random_seed", type=int, default=-1, required=False)
    parser.add_argument("--start_id", type=int, default=0, help="Start id", required=False)
    parser.add_argument("--end_id", type=int, default=0, help="End id (not included)", required=False)
    parser.add_argument("--generation_repeat_time", type=int, default=10, required=False)
    parser.add_argument("--output_folder", type=str, default='./output/', required=False)
    parser.add_argument("--see_prediction_zero_as", type=int, default=0, required=False)
    parser.add_argument("--see_correct_zero_as", type=int, default=0, required=False)
    parser.add_argument("--fix_f1", type=float, default=-1, required=False)
    parser.add_argument("--fix_f2", type=float, default=-1, required=False)
    parser.add_argument("--load_f1_f2_path", type=str, default=None, required=False)
    parser.add_argument("--get_weight_choice", type=int, default=2, required=False)
    parser.add_argument("--show_mode", type=str, default='accuracy', choices=['accuracy', 'confidence'])
    
    # parse
    args = parser.parse_args()
    
    args.output_folder=args.output_folder+args.dataset+'/'
    
    return args


args=parse_args()

repeat_time=args.generation_repeat_time

#Loading output pkl final files
file_names=[name for name in os.listdir(args.output_folder) if str(args.generation_repeat_time)+'_finish_step3.4' in name]

id_triple_list=[]
for i in range(len(file_names)):
    file_name=file_names[i].split('_')
    id_triple_list.append([i, int(file_name[0]), int(file_name[1])])
id_triple_list=sorted(id_triple_list, key=lambda x: x[1])

output=[]
current_need_start_id=args.start_id

print(id_triple_list)
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

print(len(output))#!

#Single accuracy
record_list=output   

#id_list=np.arange(len(record_list)//10)
#np.random.seed(235)
#np.random.shuffle(id_list)
#id_list=id_list[:200]
#record_list_new=[]
#for i in id_list:
#    for j in range(10):
#        record_list_new.append(record_list[i*10+j])
#record_list=record_list_new
#args.start_id=0
#args.end_id=200


if args.dataset!='prm800k':
    correct_num=0
    wrong_num=0
    for record in record_list:
        if record['predicted_result']==record['correct']:
            correct_num+=1
        else:
            wrong_num+=1
    print('Single Accuracy: ', correct_num/(correct_num+wrong_num))

#
def get_weight_1(failed_time, not_sure_time, failed_time_weight=0.5, not_sure_weight=0.2):
    s=-failed_time*failed_time_weight-not_sure_time*not_sure_weight
    s=2*np.exp(s)/(np.exp(s)+1)
    return s
def get_weight_2(failed_time, not_sure_time, failed_time_weight=0.2, not_sure_weight=0.5):
    s=failed_time_weight**failed_time*not_sure_weight**not_sure_time
    return s

get_weight_choice=args.get_weight_choice
if get_weight_choice==1:
    get_weight=get_weight_1#Consider changing this and the range
    majority_f1, majority_f2=0.0, 0.0
elif get_weight_choice==2:
    get_weight=get_weight_2#Consider changing this and the range
    majority_f1, majority_f2=1.0, 1.0


    
              
#Get type-1 and type-2 error curve
def get_predicted_dict(record_list, failed_factor=0.5, not_sure_factor=0.8):
    predicted_answer_dict={}
    for record in record_list:
        predicted_answer=record['predicted_result']
        failed_time=0
        not_sure_time=0
        for step_record in record['step_record']:
            if step_record['verification_result']==-1:
                failed_time+=1
            elif step_record['verification_result']==0:
                not_sure_time+=1
        
                
        if predicted_answer not in predicted_answer_dict:
            predicted_answer_dict[predicted_answer]=[]
            
        predicted_answer_dict[predicted_answer].append(get_weight(failed_time, not_sure_time, failed_factor, not_sure_factor))
    return predicted_answer_dict


def analyze_error(f1, f2, threshold=0.1):
    cc=0
    cw=0
    wc=0
    ww=0
    for i in range(args.end_id-args.start_id):
        predicted_answer_dict=get_predicted_dict(record_list[i*repeat_time:(i+1)*repeat_time], f1, f2)
        real_result=record_list[i*repeat_time]['correct']
        for answer in predicted_answer_dict:
            if answer==real_result:
                for score in predicted_answer_dict[answer]:
                    if score>threshold:
                        cc+=1
                    else:
                        cw+=1
            else:
                for score in predicted_answer_dict[answer]:
                    if score>threshold:
                        wc+=1
                    else:
                        ww+=1
    if cc==0:
        cc=1e-8
    if cw==0:
        cw=1e-8
    if wc==0:
        wc=1e-8
    if ww==0:
        ww=1e-8    
    return cc/(cc+cw), cc/(cc+wc), ww/(cw+ww), ww/(wc+ww), (ww+cc)/(cc+cw+wc+ww), (cc/(cc+cw)+ww/(wc+ww))/2


if args.dataset!='prm800k':
    if not args.load_f1_f2_path:
        verification_acc_list=[]
        parameter_list=[]
        for i in range(11):
            for j in range(0, i):
                for t in range(11):
                    verification_acc_list.append(analyze_error(i/10, j/10, t/10)[5])
                    parameter_list.append([i/10, j/10, t/10])
        print('Best reasoning verification acc: ',np.max(verification_acc_list))  
    
        f1, f2, t=parameter_list[np.argmax(verification_acc_list)]
    
        if (not args.load_f1_f2_path) and (args.fix_f1<=0):
            pkl.dump([f1, f2, t], open(args.output_folder+'verification_f1_f2.pkl', 'wb'))
    else:
        f1, f2, t=pkl.load(open(args.load_f1_f2_path, 'rb'))
        print('Reasoning verification acc: ',analyze_error(f1, f2, t)[5])  
    
    results_list=[]
    for t in range(101):
        results_list.append(analyze_error(f1, f2, t/100))
    
    #Plot real correct/real wrong
    fig = plt.figure(frameon=False, figsize=(10,7))
    if args.show_mode=='accuracy':
        a1=[results[0] for results in results_list]
        plt.plot(np.arange(0, 101)/100, a1, label='pred $+$ in real $+$')
        a2=[results[3] for results in results_list]
        plt.plot(np.arange(0, 101)/100, a2, label='pred $-$ in real $-$')
    elif args.show_mode=='confidence':
        a1=[results[1] for results in results_list]
        plt.plot(np.arange(0, 101)/100, a1, label=r'real $+$ in pred $+$')
        a2=[results[2] for results in results_list]
        plt.plot(np.arange(0, 101)/100, a2, label=r'real $-$ in pred $-$')
    print(np.max(a1), np.max(a2), np.min(a1), np.min(a2))
    
    
    plt.yticks([0.00, 0.20, 0.40, 0.60, 0.80, 1.00], ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'], fontsize=30)
    plt.xticks([])
    if args.show_mode=='accuracy':
        plt.ylabel('Accuracy', fontsize=30)
    else:
        plt.ylabel('Condidence', fontsize=30)
    plt.legend(loc='best', fontsize=30)
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=35)
    plt.xlabel(r'Threshold $t$', fontsize=35)
    fig.savefig(args.output_folder+'verification_img_'+args.dataset+'_'+args.show_mode+'.pdf', bbox_inches = 'tight', pad_inches = 0) #!
    
    if args.show_mode=='confidence':
        exit()
    #Plot mean
    fig = plt.figure(frameon=False, figsize=(10,4))
    a=[results[5] for results in results_list]
    plt.plot(np.arange(0, 101)/100, a, label='mean', color='r')
    
    
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=35)
    plt.ylabel('Mean acc', fontsize=30)
    plt.yticks([0.50, 0.55, 0.60, 0.65], fontsize=30)
    plt.axis([-0.05, 1.05, 0.49, 0.66])
    plt.xlabel(r'Threshold $t$', fontsize=35)
    fig.savefig(args.output_folder+'verification_img_mean_'+args.dataset+'.pdf', bbox_inches = 'tight', pad_inches = 0) #!
    
    
    
if args.dataset=='prm800k':
    correct_num=0
    wrong_num=0
    cc=0
    cw=0
    wc=0
    ww=0
    
    result_dict={}
    s=0
    
    for record in record_list:
        i=0
        for step_record in record['step_record']:
            i+=1
            human_result=step_record['human_verification_result']
            predicted_result=step_record['verification_result']
            #print(human_result, predicted_result)
            if True:
                if human_result==0:
                    human_result=args.see_correct_zero_as
                if predicted_result==0:
                    predicted_result=args.see_prediction_zero_as
            if human_result is None:
                continue
            if human_result==predicted_result:
                correct_num+=1
            else:
                wrong_num+=1
            if human_result==1 and predicted_result==1:
                cc+=1
            elif human_result==1 and predicted_result==-1:
                cw+=1
            elif human_result==-1 and predicted_result==1:
                wc+=1
            elif human_result==-1 and predicted_result==-1:
                ww+=1
            
            key=str(human_result)+'_'+str(predicted_result)
            if key not in result_dict:
                result_dict[key]=0
            result_dict[key]+=1
            s+=1


    all_num=correct_num+wrong_num
    print('Step verification acc: ',correct_num/all_num)

    ave_acc=(cc/(cc+cw)+ww/(wc+ww))/2
    print('cc:{}, cw:{}, wc:{}, ww:{}, ave acc:{}'.format(cc/all_num, cw/all_num, wc/all_num, ww/all_num, ave_acc))
    
    for key in result_dict:
        result_dict[key]/=s
    print(result_dict)
    

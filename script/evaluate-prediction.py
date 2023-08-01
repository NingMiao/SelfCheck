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
    parser.add_argument("--get_weight_choice", type=int, default=1, required=False)
    
    # parse
    args = parser.parse_args()
    
    args.output_folder=args.output_folder+args.dataset+'/'
    
    return args


args=parse_args()


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

repeat_time=args.generation_repeat_time
def collect_prediction_and_verification(record_list, failed_factor=0.5, not_sure_factor=0.8):
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
            predicted_answer_dict[predicted_answer]={'num':0, 'weight':0.0}
            
        #predicted_answer_dict[predicted_answer]+=failed_factor**failed_time*not_sure_factor**not_sure_time
        predicted_answer_dict[predicted_answer]['weight']+=get_weight(failed_time, not_sure_time, failed_factor, not_sure_factor)
        predicted_answer_dict[predicted_answer]['num']+=1
        #predicted_answer_dict[predicted_answer]+=np.random.random()
    if None in predicted_answer_dict:
        del(predicted_answer_dict[None])
    
    
    nums=[predicted_answer_dict[key]['num'] for key in predicted_answer_dict]
    try:
        max_num=np.max(nums)
    except:
        max_num=1#?there might be some problem in mathqa
    
    for key in predicted_answer_dict:
        if predicted_answer_dict[key]['num']>=max_num*0:#?
            predicted_answer_dict[key]=predicted_answer_dict[key]['weight']
        else:
            #predicted_answer_dict[key]=predicted_answer_dict[key]['weight']
            predicted_answer_dict[key]=0#?
            #Not working for weight 1
                
    
    if len(predicted_answer_dict)>0:
        keys=list(predicted_answer_dict.keys())
        final_keys=[keys[0]]
        for key in keys:
            if predicted_answer_dict[key]==predicted_answer_dict[final_keys[0]]:
                if key not in final_keys:
                    final_keys.append(key)
            elif predicted_answer_dict[key]>predicted_answer_dict[final_keys[0]]:
                final_keys=[key]
        return final_keys, predicted_answer_dict, max_num
    else:
        return None, predicted_answer_dict, max_num

def get_acc(f1, f2, voting_num=10, return_output_labels=False):
    predicted_results_list=[]
    predicted_answer_dict_list=[]
    max_num_list=[]
    for i in range(args.end_id-args.start_id):
        results, predicted_answer_dict, max_num=collect_prediction_and_verification(record_list[i*repeat_time:(i)*repeat_time+voting_num], f1, f2)
        predicted_results_list.append(results)
        predicted_answer_dict_list.append(predicted_answer_dict)
        max_num_list.append(max_num)
    
    real_result_list=[]
    for i in range(args.end_id-args.start_id):
        real_result_list.append(record_list[i*repeat_time]['correct'])
    
    correct_num=0
    no_in_list_num=0
    output_labels=[]
    for i in range(args.end_id-args.start_id):
        predicted_results=predicted_results_list[i]
        correct_result=real_result_list[i]
        if predicted_results_list[i] is not None and correct_result in predicted_results:
            correct_num+=1/len(predicted_results)
            output_labels.append(1/len(predicted_results))
        elif correct_result not in predicted_answer_dict_list[i]:
            no_in_list_num+=1
            output_labels.append(0)
        else:
            output_labels.append(0)
    
    acc=correct_num/len(real_result_list)
    
    if return_output_labels:
        return acc, output_labels
    else:
        return acc

if args.dataset!='prm800k':
    our_list=[]
    baseline_list=[]
    
    if args.load_f1_f2_path:
        load_f1_f2=pkl.load(open(args.load_f1_f2_path,'rb'))
    else:
        load_f1_f2=None
    
    best_parameter_dict={}
    our_output_labels=[]
    baseline_output_labels=[]
    for k in range(1, args.generation_repeat_time+1):
    #for k in range(10, 11):
        acc_list=[]
        parameter_list=[]
        
        if load_f1_f2:
            f1, f2=load_f1_f2[k]
            acc=get_acc(f1, f2, k)
            parameter_list.append([f1, f2])
        elif args.fix_f1>=0 and args.fix_f2>=0:
            f1, f2=args.fix_f1, args.fix_f2
            acc=get_acc(f1, f2, k)
            parameter_list.append([f1, f2])
        else:
            for i in range(21):
                f1=i/10
                if get_weight_choice==1:
                    j_start=0
                    j_end=i+1
                elif get_weight_choice==2:
                    j_start=i
                    j_end=11
                
                for j in range(j_start, j_end):
                    f2=j/10
                    parameter_list.append([f1, f2])
                    acc_list.append(get_acc(f1, f2, k))
            
            acc=np.max(acc_list)
            best_parameter=parameter_list[np.argmax(acc_list)]
            best_parameter_dict[k]=best_parameter
            f1, f2=best_parameter
            print(parameter_list[np.argmax(acc_list)])
        our_output_labels.append(get_acc(f1, f2, k, True)[1])
        baseline_result, baseline_output_labels_= get_acc(majority_f1, majority_f2, k, True)
        print('repeat_num: {}, ours:{}, majority voting acc:{}'.format(k, acc, baseline_result))
        our_list.append(acc)
        baseline_list.append(baseline_result)
        baseline_output_labels.append(baseline_output_labels_)
    
    if (not args.load_f1_f2_path) and (args.fix_f1<=0):
        pkl.dump(best_parameter_dict, open(args.output_folder+'f1_f2.pkl', 'wb'))
    
    data_len=args.end_id-args.start_id
    
    
    our_SE=[]
    for k in range(args.generation_repeat_time):
        our_SE.append(np.std(our_output_labels[k])/data_len**0.5)
    baseline_SE=[]
    for k in range(args.generation_repeat_time):
        baseline_SE.append(np.std(baseline_output_labels[k])/data_len**0.5)
    #Print accuracy
    fig = plt.figure(frameon=False, figsize=(10,7))
    plt.plot(np.arange(1, len(baseline_list)+1), baseline_list, label='majority voting')
    plt.plot(np.arange(1, len(baseline_list)+1), our_list, label='ours')
    y_range=0.18
    stride=0.02
    y_min=np.min(baseline_list+our_list)
    y_max=np.max(baseline_list+our_list)
    y_range_min=np.floor(y_min//stride)*stride
    y_range_max=y_range_min+y_range
    y_ticks=[]
    ii=0
    while True:
        if y_range_min+stride*ii<=y_range_max:
            y_ticks.append(y_range_min+stride*ii)
            ii+=1
        else:
            break
    plt.axis([0.5,10.5, y_range_min-0.005, y_range_max+0.005])
    #plt.xticks(np.arange(1,11), fontsize=15)
    plt.xticks([])
    plt.yticks(y_ticks, fontsize=30)
    plt.legend(loc='lower right', fontsize=30)
    #plt.xlabel('#Samples per question', fontsize=15)
    plt.ylabel('Accuracy', fontsize=30)
    fig.savefig(args.output_folder+'acc_img_'+args.dataset+'.pdf', bbox_inches = 'tight', pad_inches = 0) #!
    
    #Plot accuracy diff
    fig = plt.figure(frameon=False, figsize=(10,4))
    diff_list=[our_list[i]-baseline_list[i] for i in range(len(baseline_list))]
    
    diff_SE=[]
    for k in range(args.generation_repeat_time):
        print(len(our_output_labels[k]), len(baseline_output_labels[k]))
        diff_labels=[our_output_labels[k][i]-baseline_output_labels[k][i] for i in range(len(our_output_labels[k]))]
        diff_SE.append(np.std(diff_labels)/data_len**0.5)
    plt.errorbar(np.arange(1, len(baseline_list)+1), diff_list, yerr=diff_SE, color='r')
    #plt.plot(np.arange(1, len(baseline_list)+1), diff_list, color='r')
    y_range=0.04
    stride=0.01
    y_min=np.min(diff_list)
    y_max=np.max(diff_list)
    y_range_min=np.floor(y_min//stride)*stride
    y_range_max=y_range_min+y_range
    y_ticks=[]
    ii=0
    while True:
        if y_range_min+stride*ii<=y_range_max:
            y_ticks.append(y_range_min+stride*ii)
            ii+=1
        else:
            break
    plt.axis([0.5,10.5, y_range_min-0.005, y_range_max+0.005])
    plt.xticks(np.arange(1,11), fontsize=35)
    plt.yticks(y_ticks, fontsize=30)
    #plt.legend(loc='lower right', fontsize=15)
    plt.xlabel('#Samples per question', fontsize=35)
    plt.ylabel(r'$\Delta$ Accuracy', fontsize=30)
    fig.savefig(args.output_folder+'acc_diff_img_'+args.dataset+'.pdf', bbox_inches = 'tight', pad_inches = 0) #!
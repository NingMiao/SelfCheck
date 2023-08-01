import numpy as np
import json

def load_dataset(name='GSM8K', path='data/', phase=2):
    if name=='GSM8K':
        return load_GSM8K(path=path)
    if name=='MathQA':
        return load_MathQA(path=path)
    if name=='Math':
        return load_Math(path=path)
    if name=='prm800k':
        return load_prm800k(path=path, phase=phase)
    
def load_GSM8K(path='data/'):

    #GSM8K_train=[]
    #with open(path+'GSM8K/train_socratic.jsonl.txt') as f:
    #    for line in f:
    #        line=json.loads(line)
    #        answers=line['answer'].split('#### ')
    #        line['answer']=answers[1]
    #        #line['reasoning']=answers[0].strip()
    #        GSM8K_train.append(line)
    
    GSM8K_test=[]
    with open(path+'GSM8K/test.jsonl.txt') as f:
        for line in f:
            line=json.loads(line)
            answers=line['answer'].split('#### ')
            line['answer']=answers[1]
            #line['reasoning']=answers[0].strip()
            GSM8K_test.append(line)

    #return {'train': GSM8K_train, 'test': GSM8K_test}
    return {'test': GSM8K_test}
    
    

def load_MathQA(path='data/'):
    #No shuffling yet
    #MathQA_train=json.load(open(path+'MathQA/train.json'))
    #MathQA_dev=json.load(open(path+'MathQA/dev.json'))
    MathQA_test=json.load(open(path+'MathQA/test.json'))
    #MathQA_challenge_test=json.load(open(path+'MathQA/challenge_test.json'))
    
    #Each subset is a list of dicts with key ('Problem', 'Rationale', 'options', 'correct', 'annotated_formula', 'linear_formula', 'category')
    #return {'train': MathQA_train, 'dev': MathQA_dev, 'test': MathQA_test, 'challenge_test': MathQA_challenge_test}
    
    def clean(data_line):
        data={}
        data['question']=data_line['Problem']
        data['options']=data_line['options']
        data['answer']=data_line['correct']
        return data
    
    return {'test': [clean(data_line) for data_line in MathQA_test]}

def load_Math(path='data/'):
    Math_test=json.load(open(path+'MATH/MATH_np.json'))
    print('This is a subset of the testset, not the standard one!')
    def clean(data_line):
        data={}
        data['question']=data_line['question']
        data['answer']=data_line['final_answer']
        return data
    
    
    return {'test': [clean(data_line) for data_line in Math_test]}

def load_prm800k(path='data/', phase=2):
    file_path=path+'prm800k/phase'+str(phase)+'_test.jsonl'
    with open(file_path, 'r') as json_file:
        json_list = list(json_file)

    prm_test=[]
    for json_str in json_list:
        result = json.loads(json_str)
        prm_test.append(result)
    
    def clean(data_line, phase=2):
        data={}
        data['question']=data_line['question']['problem']
        
        if phase==2:
            data['steps']=[]
            data['verification_result']=[]
            for item in data_line['label']['steps']:
                data['steps'].append(item['completions'][0]['text'])
                data['verification_result'].append(item['completions'][0]['rating'])
        elif phase==1:
            data['steps']=[]
            for item in data_line['label']['steps']:
                step={}
                if item['human_completion'] is not None:
                    step['main_step'] = item['human_completion']['text']
                else:
                    chosen_id=item['chosen_completion']
                    if chosen_id is None:
                        chosen_id=0
                    try:
                        step['main_step'] = item['completions'][chosen_id]['text']
                    except:
                        continue
                step['candidates']=[]
                step['verification_result']=[]
                for itemitem in item['completions']:
                    step['candidates'].append(itemitem['text'])
                    step['verification_result'].append(itemitem['rating'])
                data['steps'].append(step)
                    
            
        return data
    
    
    
    
    return {'test': [clean(data_line, phase=phase) for data_line in prm_test]}

if __name__=='__main__':
    data = load_dataset('prm800k', phase=1)
    print(data['test'][0], len(data['test']))
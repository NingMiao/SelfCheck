#GPT-3 api
import os
import openai
from time import time, sleep
import numpy as np
import pickle as pkl
import tiktoken
openai.api_key = ''


model_dict={
    'gpt-3':{'name': 'text-davinci-003', 'req_per_min':3500, 'tok_per_min':350000, 'max_token_per_req': 4096, 'mode': 'text'}, 
    #'code':{'name': 'code-davinci-002', 'req_per_min':20, 'tok_per_min':40000, 'max_token_per_req': 8000}},
    'code':{'name': 'code-davinci-002', 'req_per_min':10, 'tok_per_min':40000, 'max_token_per_req': 8000, 'mode': 'text'},
    'gpt-3.5-0301':{'name': 'gpt-3.5-turbo-0301', 'req_per_min':3500, 'tok_per_min':80000, 'max_token_per_req': 4096, 'mode': 'chat'},
    'gpt-3.5-0613':{'name': 'gpt-3.5-turbo-0613', 'req_per_min':3000, 'tok_per_min':250000, 'max_token_per_req': 4096, 'mode': 'chat'},
    'gpt-4':{'name': 'gpt-4', 'req_per_min':3500, 'tok_per_min':90000, 'max_token_per_req': 8192, 'mode': 'chat'}}

#model_choice='gpt-3.5' #Use 'code' for free now.

#Initialize gpt api
def get_response_text(input, model_name='code-davinci-002', temp=1.0, n=1, logprobs=5, max_tokens=256, echo=False, stop='null'):
    #Input can be both string or a list of string for batch parallel
    response = openai.Completion.create(
        model=model_name,
        prompt=input,
        temperature=temp,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=n,
        logprobs=logprobs,
        echo=echo,
        stop=stop,
        )
    return response['choices'][0]['text']
    #return response

def get_response_chat(input, model_name='gpt-3.5-turbo', temp=1.0, n=1, logprobs=5, max_tokens=256, echo=False, stop='\n'):
    #echo, logprobs not working
    messages=[]
    messages.append({"role": "system", "content": "Follow the given examples and answer the question."})
    messages.append({"role": "user", "content": input})
    
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=messages,
        temperature=temp,
        n=n,
        max_tokens=max_tokens,
        stop=stop,
    )
    
    return response['choices'][0]['message']['content']


class GPT:
    def __init__(self, model_choice='gpt-3.5'):
        model=model_dict[model_choice]
        self.time_gap=0.0
        self.t=-1
        self.model=model
        
        self.last_update_time=time()
        self.limit_margin=0.8
        self.available_request_capacity=0
        self.max_requests_per_minute=self.model['req_per_min']*self.limit_margin
        self.available_token_capacity=0
        self.max_tokens_per_minute=self.model['tok_per_min']*self.limit_margin
        
        self.wait_time_base=max(0.2, 60.0/self.max_requests_per_minute*2)
        self.wait_time_multiplier=1.0
        
        self.encoding = tiktoken.get_encoding('cl100k_base')
        self.max_trial=100
        self.increase_trial=10
        
        self.sleep_until=None
        
        if self.model['mode']=='text':
            self.get_response=get_response_text
        else:
            self.get_response=get_response_chat
        
        print('model: ', self.model['name'])
        
        
    
    def __call__(self, input, temp=1.0, n=1, logprobs=1, max_tokens=256, echo=False, stop='null'):
        
        if type(input)==type([]):
            #Only support n=1
            prompt_token=sum([len(self.encoding.encode(prompt)) for prompt in input])
            response_token=len(input)*max_tokens
        else:
            prompt_token=len(self.encoding.encode(input))
            response_token=max_tokens
        token_usage=prompt_token+response_token
        
        if token_usage>self.model['max_token_per_req']:
            print('Error: max_token_per_req exceeded!')
            return None
        
        
        for i in range(self.max_trial):
            
            while True:
                current_time = time()
                if self.sleep_until is not None and self.sleep_until>current_time:
                    sleep_time=self.sleep_until-current_time
                    sleep(min(sleep_time*1.1, sleep_time+0.1)) 
                    continue
            
                seconds_since_update = current_time - self.last_update_time
                self.available_request_capacity = min(self.available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0, self.max_requests_per_minute)
                self.available_token_capacity = min(self.available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0, self.max_tokens_per_minute)    
        
                if self.available_request_capacity<1 or self.available_token_capacity<token_usage:
                    wait_time_needed=max((1-self.available_request_capacity) / self.max_requests_per_minute * 60.0, (token_usage-self.available_token_capacity) / self.max_tokens_per_minute * 60.0)
                    #print('Sleep {}s actively.'.format(wait_time_needed))
                    sleep(wait_time_needed)
                    continue
                else:
                    break
            
            
            try:
                self.last_update_time = time()
                self.available_request_capacity-=1
                self.available_token_capacity-=token_usage
                response = self.get_response(input, model_name=self.model['name'], temp=temp, n=n, logprobs=logprobs, max_tokens=max_tokens, echo=echo, stop=stop) 
                self.wait_time_multiplier=1.0
                
                if type(input)==type([]):
                    generated_texts = [""] * len(input)
                    for choice in response.choices:
                        generated_texts[choice.index] = choice.text
                    return generated_texts
                else:
                    #generated_text = response['choices'][0]['text']
                    generated_text = response
                    return generated_text
                
            except Exception as error:
                self.available_request_capacity=0
                self.available_token_capacity=0
                self.last_update_time = time()
                sleep_time=self.wait_time_base*self.wait_time_multiplier
                
                
                
                self.sleep_until=time()+sleep_time
                self.wait_time_multiplier=min(self.wait_time_multiplier*4, 2**self.increase_trial)
                print('Sleep {}s because of error: {}.'.format(sleep_time, error))
        
        return None

                
if __name__=='__main__':                
    gpt=GPT()
    input='''Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
    A:'''
    t=time()
    for i in range(0):
        output=gpt(input, temp=1.0, n=1, logprobs=1, max_tokens=200, stop='\n')
        print(output)
        print(time()-t)
        t=time()
        
    #result=openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[[
    #        {"role": "system", "content": "You are a helpful assistant."},
    #        {"role": "user", "content": "Who won the world series in 2020?"}
    #     ]]
    #     )
    #print(result)
    def run_gpt(input, stop, max_tokens, temp):
        return gpt(input, stop=stop, max_tokens=max_tokens, temp=temp)
    from src.multiprocess import multiprocess
    out=multiprocess(run_gpt, 3, [['ee', None,10,0.1]]*10)
    print(out)

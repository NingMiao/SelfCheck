import requests
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import numpy as np
from ipywidgets import IntProgress
from IPython.display import display
from tqdm import tqdm

def check(f, *args, **kwargs):
    try:
        output=f(*args, **kwargs)
        return output
    except:
        return None
                   

def multiprocess_old(f, max_worker, parameter_list):
    output_list=[None]*len(parameter_list)
    
    IP = IntProgress(min=0, max=len(parameter_list)//max_worker*2+1) # instantiate the bar
    display(IP)
    
    for j in range(len(parameter_list)//max_worker*2+1):
        IP.value += 1
        current_processing_id_list=[]
        for i in range(len(output_list)):
            if output_list[i] is None:
                current_processing_id_list.append(i)
            if len(current_processing_id_list)>=max_worker:
                break
        
        if len(current_processing_id_list)<=0:
            break
        
        threads= []
        with ThreadPoolExecutor(max_workers=len(current_processing_id_list)) as executor:
            for i in range(len(current_processing_id_list)):
                id=current_processing_id_list[i]
                threads.append(executor.submit(check, f, *parameter_list[id]))
                time.sleep(np.random.random()/100)
            
            for i in range(len(threads)):
                result=threads[i].result()
                id=current_processing_id_list[i]
                output_list[id]=result
    return output_list

def multiprocess(f, max_worker, parameter_list, progress_bar='IntProgress'):
    output_list=[None]*len(parameter_list)
    
    if progress_bar=='progress_bar':
        IP = IntProgress(min=0, max=len(parameter_list)) # instantiate the bar
        display(IP)
    else:
        pbar = tqdm(total=len(parameter_list))
    
    if True:
        
        current_processing_id_list=[]
        for i in range(len(output_list)):
            if output_list[i] is None:
                current_processing_id_list.append(i)
        
        threads= []
        with ThreadPoolExecutor(max_workers=max_worker) as executor:
            for i in range(len(current_processing_id_list)):
                id=current_processing_id_list[i]
                threads.append(executor.submit(check, f, *parameter_list[id]))
                time.sleep(np.random.random()/100)
                
            for i in range(len(threads)):
                result=threads[i].result()
                if progress_bar=='progress_bar':
                    IP.value += 1
                else:
                    pbar.update(1)
                id=current_processing_id_list[i]
                output_list[id]=result
        
        if progress_bar!='progress_bar':
            pbar.close()
    return output_list

if __name__=='__main__':
    def f(a, b):
        import numpy as np
        if np.random.random()>0.5:
            return a
        else:
            return c
    parameter_list=[[1,2],[3,4],[5,6]]
    print(multiprocess(f, 2, parameter_list))
    
    class A:
        def __init__(self):
            self.a=1
        def __call__(self, i=1):
            self.a+=1
            return self.a
    AA=A()
    parameter_list=[[1],[2],[3]]
    print(multiprocess(AA, 3, parameter_list))
    
    def g(i):
        print('start', i, time.time())
        time.sleep(np.random.random()*10)
        print('finish', i, time.time())
        return 0
    parameter_list=[[1],[2],[3]]
    multiprocess(g, 2, parameter_list)
        
    
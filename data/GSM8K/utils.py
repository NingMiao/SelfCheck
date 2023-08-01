def get_number(s):
    s_out=''
    flag=0
    for i in range(len(s)):
        if flag==0 and s[i].isdigit():
            s_out+=s[i]
            flag=1
        elif flag==1 and s[i].isdigit():
            s_out+=s[i]
        elif flag==1 and s[i]=='.':
            s_out+=s[i]
            flag=2
        elif flag==2 and s[i].isdigit():
            s_out+=s[i]
            flag=3
        elif flag==3 and s[i].isdigit():
            s_out+=s[i]
        elif flag in [1,2,3] and not s[i].isdigit():
            break    
    if flag==2:
        s_out=s_out[:-1]
    
    return s_out

def get_answer(output):
    
    result_pre=output.lower().split('so the answer is')
    if len(result_pre)==2:
        predicted_answer=get_number(result_pre[1])
    else:
        predicted_answer=''   
    
    return predicted_answer
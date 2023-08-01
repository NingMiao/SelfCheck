def get_string_between_bracket(line, left='{', right='}'):
    pos_left=-1
    pos_list=[]
    for i in range(len(line)):
        x=line[i]
        if x==left:
            pos_left=i
        if x==right:
            if pos_left>=0:
                pos_right=i
                pos_list.append([pos_left, pos_right])
                pos_left=-1
    string_list=[]
    for pos_left, pos_right in pos_list:
        string_list.append(line[pos_left+1: pos_right])
    return string_list, pos_list

def get_last_expression(line):
    line=line[:-1]
    ind=-1
    for i in range(1, len(line)+1):
        ind=-i
        if line[ind]=='{':
            break
    return line[ind+1:]
        


#Get clean expression to feed into python
operation_list=['+', '-', '*', '/', '^','÷']
digits=['0','1','2','3','4','5','6','7','8','9','.']
brackets=['(', ')']
acceptable_list=operation_list+digits+brackets

non_unit_list=[' ', '=']

def get_unit(string):
    for i in range(len(string)):
        if string[i] not in acceptable_list and string[i] not in non_unit_list:
            unit=string[i]
            if i==0:
                place='left'
            elif i==len(string)-1:
                place='right'
            elif string[i-1].isdigit():
                place='right'
            elif string[i+1].isdigit():
                place='left'
            else:
                place=''
            if place!='':
                return unit, place
    return '',''
                    
    
def run_expression(string):
    unit, unit_place=get_unit(string)
    new_string=''
    for i in range(len(string)):
        x=string[i]
        if x in acceptable_list:
            if x=='^':
                x='**'
            if x=='÷':
                x='/'
            if i-1>=0 and i<len(string)-2:
                if string[i-1]==' ' and string[i+1]==' ' and x=='x':
                    x='*'
            new_string+=x
       
    try:
        global y
        exec('y='+new_string, globals())
        if y==int(y):
            y=int(y)
        if unit!='':
            if unit_place=='left':
                return unit+str(y)
            else:
                return str(y)+unit
        else:
            return str(y)
    except:
        return ''

    
    
def get_last_expression_insert(line):
    string_list, pos_list=get_string_between_bracket(line)
    if len(string_list)==0:
        return '', []
    else:
        return string_list[-1]


def replace_with_calculator_results(line, result, force=False):
    string_list, pos_list=get_string_between_bracket(line)
    if len(string_list)==0 or (result=='' and not force):
        return line
    
    digit_start=-1
    digit_end=-1
    word_start=-1
    for i in range(pos_list[-1][1]+1, len(line)):
        s=line[i]
        if s.isspace():
            continue
        if s.isdigit():
            if digit_start==-1:
                digit_start=i
            digit_end=i                
        if s.isalpha() or s in ['"', '>']:
            word_start=i
            break
            
    if word_start==-1:
        return line[:pos_list[-1][1]+1]+' '+result
    else:
        return line[:pos_list[-1][1]+1]+' '+result+' '+line[word_start:]
        
    
if __name__=='__main__':
    #y=run_expression('100% - 50% - 75% =')
    #print(y)
    #string='so he get {10 + 40 =}'
    #print(get_last_expression(string))
    line='<{100% - 50% - 75% =} 10\n\n>'
    string=get_last_expression_insert(line)
    result=run_expression(string)
    print(string)
    print(replace_with_calculator_results(line, result))
    print(string)
def verify_middle(string):
    strings=string.split(', ')
    for item in strings:
        if item[0]!='\'' or item[-1]!='\'':
            return False
    return True

def verify_format(string):
    if string.startswith('We know that'):
        s1='We know that ['
        s2=']. So we can conclude <'
        s3='>.'
        
        s1_index_start=0
        s1_index_end=len(s1)
        if s2 not in string or s3 not in string:
            return 0, False
        
        s2_index_start=string.index(s2)
        s2_index_end=s2_index_start+len(s2)
        s3_index_start=string.index(s3)
        
        middle_1=string[s1_index_end: s2_index_start]
        middle_2=string[s2_index_end: s3_index_start]
        if (not verify_middle(middle_1)) or (not verify_middle(middle_2)):
            return 0, False
        
        return 0, True
        
    if string.startswith('The question is asking'):
        s1='The question is asking ['
        s2=']. We know that ['
        s3=']. So the answer is'
        
        s1_index_start=0
        s1_index_end=len(s1)
        if s2 not in string or s3 not in string:
            return 1, False
        
        s2_index_start=string.index(s2)
        s2_index_end=s2_index_start+len(s2)
        s3_index_start=string.index(s3)
        
        middle_1=string[s1_index_end: s2_index_start]
        middle_2=string[s2_index_end: s3_index_start]
        if (not verify_middle(middle_1)) or (not verify_middle(middle_2)):
            return 1, False
        
        return 1, True
    
    return 2, False

if __name__=='__main__':
    string1='''We know that ['{440 * (1 - 4.5) coins belong to Elsa', 'there are 440 coins in total']. So we can conclude <'there are {440 * 4.5 =} 1980 coins belong to Amalie'>.'''
    string2='''The question is asking ['how much is his sales for this month']. We know that ['Assume the sales is $4050']. So the answer is 4050.'''
    print(verify_format(string1))
    print(verify_format(string2))
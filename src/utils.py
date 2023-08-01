def transform2verifierformat(output):
    l=len(output)
    evidence_start_id=-1
    evidence_end_id=-1
    conclusion_start_id=-1
    conclusion_end_id=-1
    
    for i in range(l):
        if output[i]=='[':
            evidence_start_id=i
            break
    for i in range(evidence_start_id+1, l):
        if output[i]==']':
            evidence_end_id=i
            break
    for i in range(evidence_end_id+1, l):
        if output[i]=='<':
            conclusion_start_id=i
            break
    for i in range(conclusion_start_id+1, l):
        if output[i]=='>':
            conclusion_end_id=i
            break
            
    if evidence_start_id==-1 or evidence_end_id==-1 or conclusion_start_id==-1 or conclusion_end_id==-1:
        return ''
    else:
        evidence=output[evidence_start_id: evidence_end_id+1]
        conclusion=output[conclusion_start_id: conclusion_end_id+1]
        return 'can we draw the conclusion that '+conclusion+' from evidences '+evidence+'?'

if __name__=='__main__':
    output='''We know that ['On Monday Buddy has 30 baseball cards', 'On Tuesday Buddy loses half of them']. So we can conclude <'On Tuesday Buddy has {30 / 2=} 15 baseball cards'>.'''
    print(transform2verifierformat(output))

     
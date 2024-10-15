import random
import json

random.seed(0)

path='data/math/gsm8k'
with open(f'{path}/train.src','r') as f1, open(f'{path}/train.tgt','r') as f2, open(f'{path}/counter-gsm8k.json','w') as f:
    src=f1.readlines()
    tgt=f2.readlines()
    assert len(src)==len(tgt)
    datas=[]
    for i in range(len(src)):
        answer= tgt[i].replace(',', '').strip()
        data_dict={'index': i,
                   'question':src[i].strip(), 
                   'answer':answer}
        
        if int(answer) > 10:
            random_number = random.randint(-10, 10)
        else:
            random_number = random.randint(1, 10)
        data_dict['bully answer'] = [str(int(answer)+random_number)]
    
        datas.append(data_dict)
    json.dump(datas, f, indent=2)

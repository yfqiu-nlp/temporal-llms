import sys
import pandas as pd
import json

output_path=sys.argv[1]
model_name=sys.argv[2]

domains=['blogs', 'news', 'wikipedia']
acc_dict={}
overall_accuracy=0
noanswer_in_all_direction_cnt=0
hasanswer_in_any_direction_cnt=0
inconsistent_cnt=0
all_length=0
for domain in domains:
    print('domains:',domain)
    dataset = pd.read_csv("./dataset/tempeval-bi/bidirectional-original-qa-tempeval-test-"+domain+".csv")
    labels = dataset['answer'].values
    labels = [l.lower() for l in labels]

    dataset = pd.read_csv("./dataset/tempeval-bi/bidirectional-negative-qa-tempeval-test-"+domain+".csv")
    negative_labels = dataset['answer'].values
    negative_labels = [l.lower() for l in negative_labels]

    assert len(negative_labels) == len(labels)

    all_length+=len(labels)
    
    with open(output_path+"/tempeval-"+domain+"-"+model_name+"-original.out") as f:
        original_predictions = f.readlines()
    with open(output_path+"/tempeval-"+domain+"-"+model_name+"-negative.out") as f:
        negative_predictions = f.readlines()
    original_predictions = [ans.strip().lower().replace('true', 'yes').replace('false', 'no') for ans in original_predictions]
    negative_predictions = [ans.strip().lower().replace('true', 'yes').replace('false', 'no') for ans in negative_predictions]

    assert len(labels) == len(negative_predictions)
    assert len(labels) == len(original_predictions)
    
    acc_cnt=0
    
    for l,neg_l, original_p, negative_p in zip(labels,negative_labels,original_predictions, negative_predictions):
        if (l in original_p) and (neg_l in negative_p):
            acc_cnt+=1
            overall_accuracy+=1
        # count the samples with no right prediction in all directions (if one has answer and another has no, count it as non-consistent)
        if ((l not in original_p) and (l not in negative_p)) and ((neg_l not in negative_p) and (neg_l not in original_p)):
            print(l, original_p, neg_l, negative_p)
            noanswer_in_all_direction_cnt+=1
        else:
            hasanswer_in_any_direction_cnt+=1
            if ((l in original_p) and (neg_l not in negative_p)) or ((l not in original_p) and (neg_l in negative_p)):
                inconsistent_cnt+=1
        
    
    acc_dict[domain]=acc_cnt/len(original_predictions)

print(acc_dict)
print("Overall accuracy=",overall_accuracy/all_length)
acc_dict['overall']=overall_accuracy/all_length

print("All two direction has no answer=",noanswer_in_all_direction_cnt/all_length)
acc_dict['no_answer']=noanswer_in_all_direction_cnt/all_length

print("Inconsistent predictions=",inconsistent_cnt/hasanswer_in_any_direction_cnt)
acc_dict['inconsistent_predictions']=inconsistent_cnt/hasanswer_in_any_direction_cnt

with open(output_path+'/tempeval-'+model_name+'-bidirectional.json', 'w+') as f:
    json.dump(acc_dict, f)
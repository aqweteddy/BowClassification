from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import List
import pandas as pd
from collections import Counter


def top1_acc(preds: List[List[str]], grounds: List[str]):
    preds = [p[0] for p in preds]
    return accuracy_score(grounds, preds)

def hit_acc(preds: List[List[str]], grounds: List[str]):
    hit = 0
    for pred, ground in zip(preds, grounds):
        hit += sum(int(p == ground) for p in pred)
    # print(hit)
    
    return hit / len(preds)

def cls_report(preds: List[List[str]], grounds: List[str]):
    labels = list(set(grounds)) + ['其他']
    preds = [p[0] for p in preds]
    return classification_report(grounds, preds, target_names=labels)
    

def confusion_mat(preds: List[List[str]], grounds: List[str]):
    labels = list(set(grounds))
    preds = [p[0] for p in preds]
    mat = confusion_matrix(grounds, preds, labels=labels)
    df = pd.DataFrame(mat, columns=labels, index=labels)
    return df

def print_correct(sents: List[List[str]], preds: List[List[str]], grounds: List[str]):
    preds = [p[0] for p in preds]
    with open('correct.txt', 'w') as f:
        for sent, pred, ground in zip(sents, preds, grounds):
            sent = ' '.join(sent)
            if pred == ground:
                f.writelines([pred, '\t', sent, '\n\n'])

def print_incorrect(sents: List[List[str]], preds: List[List[str]], grounds: List[str]):
    preds = [p[0] for p in preds]
    with open('incorrect.txt', 'w') as f:
        for sent, pred, ground in zip(sents, preds, grounds):
            sent = ' '.join(sent)
            if pred != ground:
                f.writelines([pred, '\t', ground, '\t', sent, '\n\n'])

def print_stat(preds: List[List[str]], grounds: List[str], f):
    preds = [p[0] for p in preds]
    counter = Counter(preds)
    print(f'pred: {counter.most_common()}', file=f)
    counter = Counter(grounds)
    print(f'ground: {counter.most_common()}', file=f)


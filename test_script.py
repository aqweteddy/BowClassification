import json
import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm

from src import (BowClassifier, CategoryKeywordsExtractor, PreprocessPipes,
                 TrainPipes)
from src.metrics import (cls_report, confusion_mat, print_correct,
                         print_incorrect, print_stat, top1_acc)

parser = ArgumentParser()
parser = PreprocessPipes.add_parser(parser)
parser = CategoryKeywordsExtractor.add_parser(parser)
parser = TrainPipes.add_parser(parser)
parser.add_argument('--test_file', type=str)
parser.add_argument('--base_dir', type=str)
parser.add_argument('--only_test', action='store_true')

args = parser.parse_args()


# create log folder
if not os.path.exists(args.base_dir):
    os.mkdir(args.base_dir)

# save cofig
with open(os.path.join(args.base_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

args.cat_kw_to = os.path.join(args.base_dir, 'category.json')

if not args.only_test:
    # Preprocess
    print('preprocessing...')
    prep_pipe = PreprocessPipes(args)
    sents, labels = prep_pipe.get_result()

    # Extract categories keywords
    print('extracting categories keywords...')
    kw_extractor = CategoryKeywordsExtractor(args, sents, labels)
    cats_kws = kw_extractor.get_result()

    # train pipelines
    print('training pipelines...')
    train_pipe = TrainPipes(args, cats_kws)
    cats_kws = train_pipe.get_result()

# test
cls = BowClassifier(args.cat_kw_to)
df = pd.read_csv(args.test_file)
ground = df['label'].tolist()
preds = []
inps = []
for seg_text in tqdm(df['seg_text'].tolist()):
    seg_text = seg_text.split()
    inps.append(seg_text)
    r = list(map(lambda x: x[0], cls.predict(seg_text)))
    preds.append(r)
    if not r:
        print(seg_text)

with open(os.path.join(args.base_dir, 'result.txt'), 'w') as f:
    f.writelines([f'top1 acc: {top1_acc(preds, ground)}\n'])
    df = confusion_mat(preds, ground).to_string(col_space=8)
    f.writelines(['\nmatrix\n', f'{df}\n\n'])
    # df.to_csv('result.csv')
    print_incorrect(inps, preds, ground)
    print_correct(inps, preds, ground)
    print(cls_report(preds, ground), file=f, flush=True)
    print_stat(preds, ground, f)




from .base_pipeline import BasePipeline
import json
from gensim.models import Word2Vec
from tqdm import tqdm
from collections import Counter

class TrainPipes(BasePipeline):
    def __init__(self, args, category_keywords):
        self.category_keywords = category_keywords
        self.args = args

    def augment_keyword_by_w2v(self):
        w2v = Word2Vec.load(self.args.w2v_path)
        result = {}
        exist_set = set()
        for cat, word_pairs in tqdm(self.category_keywords.items()):
            result[cat] = word_pairs.copy()
            exist_set.update(cat, [pair[0] for pair in word_pairs])
            word_pairs = list(filter(lambda x: x[1] > 0.6, word_pairs))
            for word, weight in word_pairs:
                if word not in w2v.wv:
                    continue
                for sim_word, sim_weight in w2v.similar_by_word(word)[:self.args.w2v_topn]:
                    if sim_word not in exist_set:
                        exist_set.add(sim_word)
                        result[cat].append((sim_word, weight))
        self.category_keywords = result

    def statisic(self):
        for cat, wps in self.category_keywords.items():
            print(f'{cat} {len(wps)} {sum([wp[1] for wp in wps])}')

    def remove_intersection(self):
        # cnt_dct = {}
        cnt_dct = Counter()
        for cat, word_pairs in tqdm(self.category_keywords.items()):
            words = list(map(lambda x: x[0], word_pairs))
            cnt_dct.update(words)
            # for w in words:

            #     if w not in cnt_dct:
            #         cnt_dct[w] = 1
            #     else:
            #         cnt_dct[w] += 1

        redundant = set([w for w, cnt in cnt_dct.most_common() if cnt >
                     self.args.remove_intersection])
        old = self.category_keywords.copy()
        categroy_keywords = {}

        for cat, word_pairs in old.items():
            for word, weight in filter(lambda x: x[0] not in redundant, word_pairs):
                # if word in redundant:
                    # continue
                if cat in categroy_keywords:
                    categroy_keywords[cat].append([word, weight])
                else:
                    categroy_keywords[cat] = [[word, weight]]
        self.category_keywords = categroy_keywords

    def get_result(self):
        if self.args.w2v_path and self.args.w2v_topn != 0:
            self.augment_keyword_by_w2v()
        if self.args.remove_intersection != 0:
            self.remove_intersection()

        if self.args.cat_kw_to:
            with open(self.args.cat_kw_to, 'w') as f:
                json.dump(self.category_keywords, f,
                          ensure_ascii=False, indent=2)
        self.statisic()
        return self.category_keywords

    @staticmethod
    def add_parser(parser):
        """
        add preprocess parser
        """
        parser.add_argument('--cat_kw_to', type=str,
                            default='data/category.json')
        parser.add_argument('--w2v_path', type=str,
                            default='data/wiki.tw.model')
        parser.add_argument('--w2v_topn', type=int, default=0)
        parser.add_argument('--remove_intersection', default=0, type=int)
        return parser

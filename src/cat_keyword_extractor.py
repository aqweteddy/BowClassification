from gensim import corpora, models
from tqdm import tqdm
from typing import List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


class CategoryKeywordsExtractor:
    def __init__(self, args,
                 data: List[List[str]],
                 labels: List[str]):
        self.args = args
        self.sents = data
        self.labels = labels
        self.grp = pd.DataFrame(
            {'seg_text': data, 'label': labels}).groupby('label')
        self.category_label = set(self.labels)

    def get_result(self):
        """get category keywords dict

        Returns:
            dict: {'category1': [(word, weight: int)],
                   'category2': [word, weight: int] ...}
        """
        result = {}
        if self.args.method == 'tfidf':
            result = self.__tfidf()

        return result

    def __tfidf(self):
        result = {}
        if self.args.share_idf:
            # text = list(map(lambda x: x.split(), self.sents))
            dct = corpora.Dictionary(self.sents)

        for cat in tqdm(self.category_label):
            subset = self.grp.get_group(cat)
            sents = subset['seg_text'].tolist()
            if not self.args.share_idf:
                # text = list(map(lambda x: x.split(), sents))
                dct = corpora.Dictionary(sents)
            result[cat] = self.__extract_cat_kw(dct, sents)[
                :self.args.num_per_category]

        return result

    def __extract_cat_kw(self, dct, sents: List[List[str]]):
        corpus = [dct.doc2bow(sent) for sent in sents]
        tfidf = models.TfidfModel(corpus)
        id2token = {v: k for k, v in dct.token2id.items()}
        result = {}

        bow = [w for bow in corpus for w in bow]
        for idx, val in tfidf[bow]:
            if idx in result:
                result[idx][0] += 1
                result[idx][1] += val
            else:
                result[idx] = [1, val]
        # for bow in corpus:
        #     for idx, val in tfidf[bow]:
        #         if idx in result:
        #             result[idx][0] += 1
        #             result[idx][1] += val
        #         else:
        #             result[idx] = [1, val]
        result = {id2token[k]: v[1] / v[0] if self.args.score == 'mean' else v[1]
                  for k, v in result.items()}
        result = list(filter(lambda x: len(x[0]) > 1, result.items()))
        return sorted(result, key=lambda x: x[1], reverse=True)

    @staticmethod
    def add_parser(parser):
        # general
        parser.add_argument('--num_per_category', type=int, default=1000)
        parser.add_argument('--method', type=str, default='tfidf')

        # TF-IDF
        parser.add_argument('--share_idf', action='store_true')
        parser.add_argument('--score', type=str, default='mean')

        return parser

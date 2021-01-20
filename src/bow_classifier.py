import json
from typing import List


class CategoryKeywordsDict:
    def __init__(self, cat_kw_dct: dict) -> None:
        """
        {word: [(category, weight), (category, weight)}
        Args:
            cat_kw_dct (dict): {'category': [[kw, weight], [kw, weight] ]}

        Returns:
            None
        """
        self.dct = {}
        self.cat_sum_score = {cat: sum(map(lambda x: x[1], word_pairs)) for cat, word_pairs in cat_kw_dct.items()}
        self.category_list = list(cat_kw_dct.keys())
        for cat, kw_pairs in cat_kw_dct.items():
            for kw, weight in kw_pairs:
                if kw not in self.dct:
                    self.dct[kw] = {}    
                self.dct[kw][cat] = weight
                # self.dct[kw][cat] = 1

    
    def get_category_list(self):
        return self.category_list
    
    def get_category_sum_score(self, category: str):
        return self.cat_sum_score.get(category, -1)

    def __contains__(self, word: str):
        return word in self.dct

    def __getitem__(self, word: str):
        """return category and weight

        Args:
            word (str): word to query

        Returns:
            dict: {'category': weight, 'category': weight}
        """
        tmp = self.dct.get(word, dict())
        # return {k: w for k, w in tmp} if tmp else tmp
        return tmp
    
    def __len__(self):
        return len(self.dct)


class BowClassifier:
    def __init__(self, category_file: str, threshold: float=0.3) -> None:
        cat2keywords = self.load_category_keywords(category_file)
        self.word_dct = CategoryKeywordsDict(cat2keywords)
        self.catgory_list = self.word_dct.get_category_list()
        self.threshold = threshold
        print(self.catgory_list)
    
    @staticmethod
    def load_category_keywords(category_file:str):
        with open(category_file) as  f:
            return json.load(f)
    
    def predict(self, sent: List[str]):
        """predict

        Args:
            sent (List[str]):  tokenized sentence
        """
        result = {k: 0. for k in self.catgory_list}
        for word in sent:
            if word in self.word_dct:
                for k, v in self.word_dct[word].items():
                    result[k] += v
                    # print(k, v)
        # result = {k: v / self.word_dct.get_category_sum_score(k) * 100 for k, v in result.items()}
        tot_score = sum(result.values())

        if tot_score == 0.0:
            result = [('其他', 1.)]
        else:
            result = sorted(result.items(), key=lambda x: x[1] / tot_score, reverse=True)
            r = result[:2]
            if not r:
                r = [result[0]]
            result = r
        if len(result) == 0:
            print(sent)
        return result

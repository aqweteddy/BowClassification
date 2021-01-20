import pandas as pd
from .base_pipeline import BasePipeline

class PreprocessPipes(BasePipeline):
    def __init__(self, args):
        self.args = args
        self.data, self.labels = self.load_data(args.train_file)

    @staticmethod
    def load_data(file):
        df = pd.read_csv(file)
        label = df['label']
        text = list(map(lambda x: x.split(), df['seg_text'].tolist()))
        return text, label.tolist()

    def get_result(self):
        """return data

        Returns:
            List[List[str]]: [description]
            List[str]: labels
        """
        if self.args.stopword_file:
            self.stopword_filter()
        return self.data, self.labels

    def stopword_filter(self):
        stopword = set()
        with open(self.args.stopword_file) as f:
            for line in f.readlines():
                stopword.add(line.strip())
        self.data = [[w for w in sent if w not in stopword and len(w) > 1]
                     for sent in self.data]

    @staticmethod
    def add_parser(parser):
        """
        add preprocess parser
        """
        parser.add_argument('--train_file', type=str)
        parser.add_argument('--stopword_file', type=str, default='')
        return parser

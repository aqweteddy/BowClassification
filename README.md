# BowClassifier

* 速度快
* 

## data

* 使用 ettoday 2016 新聞資料(不公開)

## 作法

1. 以 TF-IDF 法計算每個類別($c$)的關鍵字($k_i$)和分數($p_i$)
2. 掃過每篇文章，如果某文章出現關鍵字$k_i$，則類別 $c$ 的累積分數加 $p_i$

## benchmark

* accuracy
* best: 0.786
* 同份資料 roberta 表現: 0.87

###config

```json
{
    "train_file": "data/et_2016_seg_train.csv",
    "stopword_file": "data/stopwords.txt",
    "num_per_category": 20000,
    "method": "tfidf",
    "share_idf": true,
    "score": "sum",
    "cat_kw_to": "data/category.json",
    "w2v_path": "data/wiki.tw.model",
    "w2v_topn": 0,
    "remove_intersection": 10,
    "test_file": "data/et_2016_seg_test.csv",
    "only_test": false
}
```
### num_per_category

* 5000: 0.7466296286857463
* 10000:0.7536124773821963
* 20000:0.7703305384948648
* 30000:0.778689569051199
* 40000:0.7822064782486812
* 50000: 0.7867427814164479
* 60000: 0.7867172965671908
* -1: 0.7805499630469686

## remove_intersection

* 15: 0.7678075384184102
* 10: 0.7822064782486812
* 5: 0.776931114452458


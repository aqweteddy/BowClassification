python test_script.py --train_file data/et_2016_seg_train.csv\
                      --base_dir log/tfidf_regard-one-doc_word_60000\
                      --num_per_category 60000\
                      --stopword_file data/stopwords.txt\
                      --method tfidf\
                      --score sum\
                      --remove_intersection 10\
                      --test_file data/et_2016_seg_test.csv\
                      --share_idf
                    #   --w2v_path data/wiki.tw.model\
                    #   --w2v_topn 10\
                    #   --only_test

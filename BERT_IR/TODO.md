* preprocess.py 
   * 改成args.parser
      * --is_spoken
      * --is_train
      * --is_short

* run_classifier_[new_]TDT2.py
   * 增加args.parser
      * --test_mode 
         * long,text
         * long,spoken
         * short,text
         * short,spoken
      * --train_mode
         * long,text
         * long,spoken


* 程式修改目標
   * run_classifier_TDT2.py
      * load model
      * training set/test set/輸出檔 argument

* 實驗目標
   * 傳統模型與BERT之線性組合
   * 加 feature
   * 調整character的輸入(更重要的詞彙)
   
   * [opt.] 改成 pairwise
   * [opt.] 調整BERT模型參數

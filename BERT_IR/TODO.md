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

export TASK_NAME = "TDT2"
export DATA_DIR = "data"

python3 run_classifier_new_TDT2.py --task_name TDT2 --do_train --do_eval --do_lower_case \
                                   --data_dir $DATA_DIR/$TASK_NAME --bert_model bert-base-chinese \
                                   --max_seq_length 128 --train_batch_size 32 --learning_rate 2e-5 \
                                   --num_train_epochs 3.0 --output_dir exp/TDT2_exp

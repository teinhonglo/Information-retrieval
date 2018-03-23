#!/bin/bash

ROOTDIR=`pwd`/..
DATA=`pwd`/data
EXPPATH=`pwd`/exp

if [ ! -d "$DATA" ]; then
  mkdir $DATA
  mkdir $DATA/Train
  mkdir $DATA/Test
fi

if [ ! -d "$EXPPATH" ]; then
  mkdir $EXPPATH
fi

test_qry=$ROOTDIR/Corpus/TDT2/QUERY_WDID_NEW
test_rel=$ROOTDIR/Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt
train_qry=$ROOTDIR/Corpus/TDT2/Train/XinTrainQryTDT2/QUERY_WDID_NEW
train_rel=$ROOTDIR/Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain

TRAIN_DATA=$DATA/Train/x_qry_mdl.npy

if [ ! -f $TRAIN_DATA ]; then
  python local/VSM.py --qry_dataset $test_qry --rel_dataset $test_rel --data_storage $DATA/Test --is_train False
  python local/VSM.py --qry_dataset $train_qry --rel_dataset $train_rel --data_storage $DATA/Train --is_train True
fi

MODEL_PATH=$EXPPATH/final.h5

if [ ! -f $MODEL_PATH ]; then
  python Train.py --learn_rate 0.001 --batch_size 32 --epochs 10 --num_hids 0 --embed_dim 300 --save_best_only True
fi
  
python Test.py --exp_path exp --model_name final.h5 --isTraining False

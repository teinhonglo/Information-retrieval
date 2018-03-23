#!/usr/bin/env python3
import sys, os
rootDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(rootDir)
sys.path.append(rootDir +"/../Tools")

import numpy as np
import tensorflow as tf
from keras import backend as K
''' Import keras to build a DL model '''
from keras.models import load_model

''' Setting optimizer '''
from keras.optimizers import Adam, SGD

import cPickle as pickle
from Evaluate import EvaluateModel
import argparse
from argparse import RawTextHelpFormatter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

def cosineFast(qry, qry_IDs, doc, doc_IDs):
    # cosine similarity
    result = np.argsort(-np.dot(qry, doc.T), axis = 1)
    # prepare ranking list
    qry_docs_ranking = {}
    for q_idx, q_ID in enumerate(qry_IDs):
        docs_ranking = []
        for doc_idx in result[q_idx]:
            docs_ranking.append(doc_IDs[doc_idx])
        qry_docs_ranking[q_ID] = docs_ranking
    return qry_docs_ranking 
                  
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')                  
                  
def main(args):                  
    exp_path = args["exp_path"]
    isTraining = args["isTraining"]
    model_name = args["model_name"]
    
    if isTraining:
        data_path = "data/Train"
        rel_path = "../Corpus/TDT2/Train/QDRelevanceTDT2_forHMMOutSideTrain" 
    else:
        data_path = "data/Test"
        rel_path = "../Corpus/TDT2/AssessmentTrainSet/AssessmentTrainSet.txt" 

    # Read data
    with open(data_path + "/qry_IDs.pkl", "rb") as f: qry_IDs = pickle.load(f)
    with open(data_path + "/doc_IDs.pkl", "rb") as f: doc_IDs = pickle.load(f)
    
    qry = np.load(data_path + "/x_qry_tf_mdl.npy")
    doc = np.load(data_path + "/doc_mdl.npy")
    
    # Load model
    model = load_model(exp_path + "/" + model_name)
    # Evaluation
    evaluate_model = EvaluateModel(rel_path, isTraining)

    with tf.device('/device:GPU:0'):
        # Train
        rel_mdl = model.predict(qry)

    qry_docs_ranking = cosineFast(rel_mdl, qry_IDs, doc, doc_IDs)
    mAP = evaluate_model.mAP(qry_docs_ranking)
    print(mAP)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This program runs Test.py on a prepared corpus.\n
                                                    sample argument setting is as follows:\n
                                                    python Test.py --exp_path exp --model_name final.mdl --isTraining False
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-ep', '--exp_path', help='learn rate', required=True)
    parser.add_argument('-mn', '--model_name', help='relevant dataset', required=True)
    parser.add_argument('-it', '--isTraining', type=str2bool, nargs='?', const=True, help='Steps')
    args = vars(parser.parse_args())
    main(args)
#!/usr/bin/env python3
import sys, os
rootDir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(rootDir)
sys.path.append(rootDir +"/../Tools")

import numpy as np
np.random.seed(5566)

import tensorflow as tf
from keras import backend as K
''' Import keras to build a DL model '''
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization
from keras.callbacks import ModelCheckpoint
''' Setting optimizer    '''
from keras import optimizers
import argparse
from argparse import RawTextHelpFormatter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True),
                  inter_op_parallelism_threads = 1, intra_op_parallelism_threads = 1))

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def main(args):
    optimizer = optimizers.Adam(lr=args['learn_rate'])
    batch_size = args['batch_size']
    epochs = args['epochs']
    save_best_only = args['save_best_only']
    num_hids = args['num_hids']
    embed_dim = args['embed_dim']
    exp_path = args['exp_path']
    data_path = args['data_path']
    loss = "logcosh"
    vocab_size = 51253
    split_data = 0.2

    print "Read data"
    X = np.load(data_path + "/x_qry_tf_mdl.npy")
    Y = np.load(data_path + "/y_qry_mdl.npy")
    total = X.shape[0]
    validation_split = int(total * (1 - split_data))
    tr_x, val_x = X[:validation_split,:], X[validation_split:,:]
    tr_y, val_y = Y[:validation_split,:], Y[validation_split:,:]
    '''
    mean = np.mean(X, axis = 0)
    stdv = np.std(X, axis = 0)
    np.save(exp_path + "/mean.npy", mean)
    np.save(exp_path + "/stdv.npy", stdv)
    
    valid_idx = np.nonzero(stdv)
    tr_x[:, valid_idx] = (tr_x[:, valid_idx] - mean[valid_idx]) / stdv[valid_idx]
    val_x[:, valid_idx] = (val_x[:, valid_idx] - mean[valid_idx]) / stdv[valid_idx]
    '''
    print('Building a model whose optimizer=logcosh, activation function=relu')
    model = Sequential()
    # input layer
    model.add(BatchNormalization(input_shape =(vocab_size, )))
    # hidden layer
    for _ in range(num_hids):
        model.add(Dense(embed_dim, activation="relu"))
    # output layer    
    model.add(Dense(vocab_size, activation="relu"))
    model.summary()

    # Model check point
    checkpoint = ModelCheckpoint(exp_path +"/first_dnn_" + loss + "_weights-{epoch:02d}-{val_loss:.2f}.hdf5", 
                                 monitor='val_loss', verbose=0, save_best_only=save_best_only, mode='min')
    callbacks_list = [checkpoint]
    with tf.device('/device:GPU:0'):
        # Train
        model.compile(optimizer = optimizer, loss = loss)
        model.fit(tr_x, tr_y, 
                  validation_data=(val_x, val_y),
                  epochs = epochs, 
                  verbose=1,
                  batch_size=batch_size,
                  shuffle=True
                  #validation_split=0.2
                  #callbacks=callbacks_list
                  )
        model.save(exp_path + "/final.h5")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""This program runs Train.py on a prepared corpus.\n
                                                    sample argument setting is as follows:\n
                                                    python Training --learn_rate 0.001 --batch_size 32 --epochs 20 --num_hids 0 --embed_dim 300 --save_best_only True
    """, formatter_class=RawTextHelpFormatter)
    parser.add_argument('-lr', '--learn_rate', type = np.float, help='learn rate', required=True)
    parser.add_argument('-bs', '--batch_size', type = np.int, help='relevant dataset', required=True)
    parser.add_argument('-eps', '--epochs', type = np.int, help='epochs', required=True)
    parser.add_argument('-nh', '--num_hids', type = np.int, help='number of hidden layer', required=True)
    parser.add_argument('-ed', '--embed_dim', type = np.int, help='embedded dim', required=True)
    parser.add_argument('-epth', '--exp_path', help='exp path', required=True)
    parser.add_argument('-dpth', '--data_path', help='data path', required=True)
    parser.add_argument('-sbo', '--save_best_only', type=str2bool, nargs='?', const=True, help='Steps')
    args = vars(parser.parse_args())
    main(args)
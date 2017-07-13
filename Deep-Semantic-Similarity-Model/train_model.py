import numpy as np
np.random.seed(1331)

from keras.callbacks import ModelCheckpoint
import cPickle as Pickle
import dssm_dnn_single as dssm_dnn
import random
from random import shuffle

steps_per_epoch = 180018
epochs = 50
batch_size = 54
feature_filename = str(batch_size) + "_gamma_shuffle_Spk"
train_data_path = "model/UM/"
model_name = "NN_Model/DSSM_TD_"+feature_filename+".h5"
model_weights_name = "NN_Model/DSSM_WEIGHTS_SD_"+feature_filename+".h5"
Epochs_filepath="Epochs/with_gamma_single_Spk/"+feature_filename+"_weights-{epoch:02d}-{val_loss:.2f}.hdf5"

with open("../Information-retrieval/Corpus/pseudo_qry_train_set_Spk.pkl", "rb") as tr_f : qry_train_set = Pickle.load(tr_f)
with open("../Information-retrieval/Corpus/pseudo_qry_val_set_Spk.pkl", "rb") as tr_f : val_qry_train_set = Pickle.load(tr_f)
with open(train_data_path + "query_list.pkl", "rb") as qry_lst_f: query_list = Pickle.load(qry_lst_f)
with open(train_data_path + "query_model.pkl", "rb") as qry_md_f: query_model = Pickle.load(qry_md_f)

with open(train_data_path + "test_query_list.pkl", "rb") as qry_lst_f: val_query_list = Pickle.load(qry_lst_f)
with open(train_data_path + "test_query_model.pkl", "rb") as qry_md_f: val_query_model = Pickle.load(qry_md_f)

with open(train_data_path + "doc_list_s.pkl", "rb") as doc_lst_f: doc_list = Pickle.load(doc_lst_f)
with open(train_data_path + "doc_model_s.pkl", "rb") as doc_md_f: doc_model = Pickle.load(doc_md_f)

def preprocess(qry_train_set, query_list, query_model, doc_list, doc_model):
    random.seed(1331)
    shuffle(qry_train_set)
    qry_vec = []
    d1_vec = []
    d2_vec = []
    d3_vec = []
    d4_vec = []
    answer = []
    for idx, trn in enumerate(qry_train_set):
        qry_idx = query_list.index(trn[0])
        qry_vec.append(query_model[qry_idx])
        [d1_n, d2_n, d3_n, d4_n] = trn[1:]
        d1_vec.append(doc_model[doc_list.index(d1_n)])
        d2_vec.append(doc_model[doc_list.index(d2_n)])
        d3_vec.append(doc_model[doc_list.index(d3_n)])
        d4_vec.append(doc_model[doc_list.index(d4_n)])
        answer.append(np.array([1, 0, 0, 0]))
    return [qry_vec, d1_vec, d2_vec, d3_vec, d4_vec, answer]

def generate_arrays(qry_vec, d1_vec, d2_vec, d3_vec, d4_vec, answer, batch_size):
    total_size = len(qry_vec)
    while 1:
        for i in xrange(total_size / batch_size + 1):
            st = i * batch_size
            ed = st + batch_size
            if ed >= total_size:
                ed = None
            q = np.vstack(qry_vec[st:ed])
            d1 = np.vstack(d1_vec[st:ed])
            d2 = np.vstack(d2_vec[st:ed])
            d3 = np.vstack(d3_vec[st:ed])
            d4 = np.vstack(d4_vec[st:ed])
            a = np.vstack(answer[st:ed])
            x = [q, d1, d2, d3, d4]
            y = np.array(a)
            yield (x, y)

[qry_vec, d1_vec, d2_vec, d3_vec, d4_vec, answer] =  preprocess(qry_train_set, query_list, query_model, doc_list, doc_model)
[val_qry_vec, val_d1_vec, val_d2_vec, val_d3_vec, val_d4_vec, val_answer] =  preprocess(val_qry_train_set, val_query_list, val_query_model, doc_list, doc_model)

step_per_epochs = len(qry_vec)
print step_per_epochs

val_qry_vec = np.vstack(val_qry_vec)
val_d1_vec = np.vstack(val_d1_vec)
val_d2_vec = np.vstack(val_d2_vec)
val_d3_vec = np.vstack(val_d3_vec)
val_d4_vec = np.vstack(val_d4_vec)
val_answer = np.vstack(val_answer)

model = dssm_dnn.create_model()

checkpoint = ModelCheckpoint(Epochs_filepath, monitor='val_loss', verbose=0, save_best_only=False, mode='min')
callbacks_list = [checkpoint]

model.fit_generator(generate_arrays(qry_vec, d1_vec, d2_vec, d3_vec, d4_vec, answer, batch_size), 
                    validation_data=([val_qry_vec, val_d1_vec, val_d2_vec, val_d3_vec, val_d4_vec], val_answer),
                    steps_per_epoch=(steps_per_epoch / batch_size + 1), 
                    callbacks=callbacks_list,
                    epochs = epochs)

model.save(model_name)
model.save_weights(model_weights_name)

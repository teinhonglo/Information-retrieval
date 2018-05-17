import numpy as np
np.random.seed(1331)

import cPickle as Pickle
import ProcDoc
import word2vec_model
from keras.preprocessing.sequence import pad_sequences

model_path = "../Corpus/model/TDT2/UM/"
document_path = "../Corpus/SPLIT_DOC_WDID_NEW"    
query_path = "../Corpus/Train/XinTrainQryTDT2/QUERY_WDID_NEW"
test_query_path = "../Corpus/QUERY_WDID_NEW"
non_seeing_word = {}

def content2Emb(content_model, wordVec, word_length):
    nameList = []
    EmbeddingList = []
    global non_seeing_word
    for name, content in content_model.items():
        nameList.append(name)
        curEmbedding = []
        for w in content.split():
            if w in wordVec:
                curEmbedding.append(wordVec[w])
            else:
                if w in non_seeing_word:
                    randVec = non_seeing_word[w]
                else:
                    randVec = np.random.uniform(-2.5, +2.5, word_length)
                    print w
                    non_seeing_word[w] = np.copy(randVec)
                curEmbedding.append(randVec)
        curEmbedding = np.array(curEmbedding)
        EmbeddingList.append(np.copy(curEmbedding))
    return [nameList, EmbeddingList]

def content2List(content_model, objList):
    contentList = []
    maxL = 0
    for idx, name in enumerate(objList):
        content = content_model[name].split() 
        cur_content = []
        maxlen = 1794 - len(content)
        if len(content) > maxL: maxL = len(content)
        for w_id in content:
            cur_content.append(int(w_id) + 1)
        for i in xrange(maxlen):
            cur_content.append(0)
        contentList.append(np.array(cur_content))
    #print contentList
    print maxL
    return np.asarray(contentList)

def rePermute(nameList, EmbeddingList, objList):
    newEmbeddingList = []
    for idx, name in enumerate(objList):
        cur_idx = nameList.index(name)
        print EmbeddingList[cur_idx].shape
        newEmbeddingList.append(EmbeddingList[cur_idx])
    return newEmbeddingList

with open(model_path + "doc_list.pkl", "rb") as f: doc_list = Pickle.load(f)
with open(model_path + "query_list.pkl", "rb") as f: qry_list = Pickle.load(f)
with open(model_path + "test_query_list.pkl", "rb") as f: tstQry_list = Pickle.load(f)

wordModel = word2vec_model.word2vec_model()
wordVec = wordModel.getWord2Vec()
vocab_length = wordModel.vocabulary_length
print vocab_length

# document
doc = ProcDoc.read_file(document_path)
doc = ProcDoc.doc_preprocess(doc)
#[docTmpList, docEmbList] = content2Emb(doc, wordVec, 100)
#doc_emb = rePermute(docTmpList, docEmbList, doc_list)
#doc_emb = content2List(doc, doc_list)
#doc_emb = np.asarray(doc_emb)
#print doc_emb.shape
#np.save(model_path + "doc_id_fix_pad.npy", doc_emb)

# train query
query = ProcDoc.read_file(query_path)
query = ProcDoc.query_preprocess(query)
#[qryTmpList, qryEmbList] = content2Emb(query, wordVec, 100)
#qry_emb = rePermute(qryTmpList, qryEmbList, qry_list)
qry_emb = content2List(query, qry_list)
qry_emb = np.asarray(qry_emb)
print qry_emb.shape
np.save(model_path + "qry_id_fix_local_pad.npy", qry_emb)

# test query
test_query = ProcDoc.read_file(test_query_path)
test_query = ProcDoc.query_preprocess(test_query)
#[tstQryTmpList, tstQryEmbList] = content2Emb(test_query, wordVec, 100)
#tstQry_emb = rePermute(tstQryTmpList, tstQryEmbList, tstQry_list)
tstQry_emb = content2List(test_query, tstQry_list)
tstQry_emb = np.asarray(tstQry_emb)
print tstQry_emb.shape
np.save(model_path + "tstQry_id_fix_local_pad.npy", tstQry_emb)
'''
# zero padding
qry_emd = pad_sequences(qry_emb, maxlen=2907, dtype='float32', padding="post")
doc_emd = pad_sequences(doc_emb, maxlen=2907, dtype='float32', padding="post")
tstQry_emd = pad_sequences(tstQry_emb, maxlen=2907, dtype='float32', padding="post")

print len(qry_emb[0])
print len(tstQry_emb[0])
print len(doc_emb[0])
np.save(model_path + "qry_id_fix_pad.npy", qry_emb)
np.save(model_path + "doc_id_fix_pad.npy", doc_emb)
np.save(model_path + "tstQry_id_fix_pad.npy", tstQry_emb)
'''
'''
# Create relevant and non-relevant train set
from random import shuffle
from random import randint
with open(model_path + "HMMTraingSetDict.pkl", "rb") as RelFile: HMMTraingSetDict = Pickle.load(RelFile)
train_list = []
irr_doc = []
num_of_qry = 800
num_of_doc = 2265

for qry_idx in xrange(num_of_qry):
    qry_name = qry_list[qry_idx]
    num_of_rel = len(HMMTraingSetDict[qry_name])
    for doc_name in HMMTraingSetDict[qry_name]:
        doc_idx = doc_list.index(doc_name)
        train_list.append([qry_idx, doc_idx, 1]) 
    
    while True:
        doc_idx = randint(0, 2264)
        if (doc_idx in HMMTraingSetDict[qry_name]) or (doc_idx in irr_doc):
            continue
        else:
            irr_doc.append(doc_idx)
            train_list.append([qry_idx, doc_idx, 0])
            num_of_rel -= 1
        if num_of_rel == 0:
            break
shuffle(train_list)
with open("../Corpus/rel_irrel/TDT2/pointwise_list_small.pkl", "wb") as pFile: Pickle.dump(train_list, pFile, True)
'''

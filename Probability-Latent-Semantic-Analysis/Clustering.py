import os
import sys
import numpy as np
import kmeans

class DataInfo:
    def __init__(self, ID, coor):
        self.ID = ID        # title
        self.coor = coor    # numpy array
        
    def getID(self):
        return self.ID
    
    def getCoor(self):
        return self.coor

class ClusterModel(object):
    def __init__(self, topic_dict, vocab, k = 4):
        print("create cluster model")
        self.k = k
        self.vocab_info = [vocab, len(list(vocab.keys()))]
        self.tar_list = self.__dict2dataInfo(topic_dict)
        print("K means ...")
        [self.clusters, self.centroids] = kmeans.kmeans(self.tar_list, k)
        
    def __dict2dataInfo(self, ori_dict):
        tar_list = []
        [vocab, num_vocabs] = self.vocab_info
        for o_id, o_wls in ori_dict.items():
            obj_vec = np.zeros(num_vocabs)
            for o_w, o_wc in o_wls.items():
                w_idx = vocab[o_w]
                obj_vec[w_idx] = o_wc
            data = DataInfo(o_id, np.copy(obj_vec))
            tar_list.append(data)
        return tar_list
        
    def save(self, dir_path = "Topic"):
        # p(w|z)
        clusters = self.clusters
        [vocab, num_vocabs] = self.vocab_info
        topic = np.empty([len(clusters), num_vocabs])
        with open(dir_path + "/pwd.txt", 'w') as outfile:
            outfile.write("Topic")
            # header
            for wID in list(vocab.keys()):
                outfile.write(", " + str(wID))
            outfile.write("\n")
            # content
            for c_idx, cluster in enumerate(clusters):
                pwz = np.zeros(num_vocabs)
                print(len(cluster))
                for doc_data_info in cluster:
                    pwz += doc_data_info.getCoor()
                pwz /= np.sum(pwz, axis = 0)
                topic[c_idx] = np.copy(pwz)
                outfile.write("cluster " + str(c_idx))
                # write to outfile
                for pw in pwz:
                    outfile.write(", " + str(pw))
                outfile.write("\n")
        np.save(dir_path + "/pwz_list", topic)
            

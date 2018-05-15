import pandas as pd
import numpy as np
import cPickle as pickle
import argparse
from argparse import RawTextHelpFormatter

def main(args):
    vectors_dict = {}
    vectors_path = args['path']
    df_vectors = pd.read_csv(vectors_path,header = None,delimiter="\t")
    #print(df_vectors)
    df_vectors.columns = ['origin']
    df_vectors['word'], df_vectors['array'] = df_vectors['origin'].str.split(' ', 1).str
    del df_vectors["origin"]
    for i in df_vectors.index:
        vectors_dict[df_vectors['word'][i]] = np.array(list(df_vectors['array'][i].split(' '))).astype(np.float)
    with open(vectors_path + ".pkl" ,"wb") as gfile:
        pickle.dump(vectors_dict,gfile,True)
        print("txt2npy done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="", formatter_class=RawTextHelpFormatter)
    parser.add_argument('--path', '--path', nargs='?', const=True)
    args = vars(parser.parse_args())
    main(args)
    

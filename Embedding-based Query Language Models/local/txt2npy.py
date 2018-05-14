import pandas as pd
import numpy as np
vectors_dict = {}
vectors_path = 'data/vectors.txt'
df_vectors = pd.read_csv(vectors_path,header = None,delimiter="\t")
#print(df_vectors)
df_vectors.columns = ['origin']
df_vectors['word'], df_vectors['array'] = df_vectors['origin'].str.split(' ', 1).str
del df_vectors["origin"]
for i in df_vectors.index:
    #print(df_vectors['word'][i])
    #print(np.array(list(df_vectors['array'][i].split(' '))))
    vectors_dict[df_vectors['word'][i]] = np.array(list(df_vectors['array'][i].split(' ')))
np.save("data/glove50.npy",vectors_dict) 
#vectors_npy = np.load("glove50.npy")
#print(vectors_npy)
print("txt2npy done")

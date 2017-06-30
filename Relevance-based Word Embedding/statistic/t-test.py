import numpy as np
import cPickle as Pickle
from scipy import stats

with open("result.pkl", "rb") as file: TD = Pickle.load(file)
with open("result_s.pkl", "rb") as file: SD = Pickle.load(file)
print stats.ttest_ind(SD, SD)


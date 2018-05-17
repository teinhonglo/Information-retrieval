import numpy as np
import cPickle as pickle
np.set_printoptions(threshold=np.nan)


input_np = np.eye(51253,dtype=int) 
NRM_SWM_weight_list = np.load("data/NRM_SWM_weights_list.npy")
#print(NRM_SWM_weight_list_npy[0][0])


#bottle neck
#print(NRM_SWM_weight_list[i].shape)
print("calculating bottleneck")
element_wise = np.dot(input_np,NRM_SWM_weight_list[0])
print(element_wise.shape)
btn_dict = {}
#print(len(element_wise))
for i in range(len(element_wise)):
    btn_dict[str(i)] = element_wise[i]
with open("data/nrm_swm_btn.pkl" ,"wb") as btfile:
    pickle.dump(btn_dict,btfile,True)
print("bottleneck done")
#optput
print("calculating output...")
element_wise2 = np.dot(element_wise,NRM_SWM_weight_list[1])
print(element_wise2.shape)
output_dict = {}
#print(len(element_wise))
for i in range(len(element_wise2)):
    output_dict[str(i)] = element_wise2[i]
with open("data/nrm_swm_output.pkl" ,"wb") as outputfile:
    pickle.dump(output_dict,outputfile,True)
print("output done")

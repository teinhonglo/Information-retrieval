import numpy as np
import matplotlib.pyplot as plt
import cPickle as Pickle

numOfCluster = 1
corpus = "TDT2"
method = "SWM"
spoken = ""
history_path = "NN_Model/" + corpus + "/" + method + "/" + str(numOfCluster) + "/"
train_loss = []
val_loss = []

for i in xrange(numOfCluster):
    with open("RLE_" + method + spoken + "_1_history.pkl", "rb") as f: history = Pickle.load(f)
    train_loss.append(history[0])
    val_loss.append(history[2])

''' Visualize the loss and accuracy of both models'''
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.figure()
for i in xrange(numOfCluster):
    loss_adam = train_loss[i]
    val_loss_adam = val_loss[i]
    plt.plot(range(len(loss_adam)), loss_adam, label='Train_' + str(i))
    if val_loss_adam != None:
        plt.plot(range(len(val_loss_adam)), val_loss_adam, label='Val_' + str(i))
plt.title('Loss')
plt.legend(loc='upper left')
#plt.show()
plt.savefig("train_result_" + method + "_" + str(numOfCluster) + spoken + ".png", dpi=300, format='png')



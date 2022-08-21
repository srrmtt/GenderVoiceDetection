import numpy as np
import scipy as sp
from matplotlib.pyplot import cm
# my libraries import 
import preprocessing as prep
import plot as plt

N_FEATURES = 12

if __name__ == '__main__':
    fileTR = './data/Train.txt'
    fileTE = './data/Train.txt'

    features = ["Feature(" + str(x) + ")" for x in range(N_FEATURES)]

    DTR, LTR = prep.load_dataset(fileTR)
    DTE, LTE = prep.load_dataset(fileTE)
    print(DTR.shape)
    #plt.plot_features_distr(DTR, LTR, features)
    
    DTR = prep.z_norm(DTR)
    DTE = prep.z_norm(DTE)

    ### DATA VISUALIZATION ###
    maleDTR = DTR[:, LTR == 0]
    femaleDTR = DTR[:, LTR == 1]
    #plt.plot_features_distr(DTR, LTR, features)
    #plt.plot_relation_beetween_feautures(DTR, LTR, features)
    #plt.plot_heatmap(DTR, features, cm.Greys)
    #plt.plot_heatmap(maleDTR, features, cm.Blues)
    #plt.plot_heatmap(femaleDTR, features, cm.Reds)

    



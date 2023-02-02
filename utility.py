import numpy as np
import preprocessing as prep
import pickle 
import random
from copy import deepcopy
import os

def vcol(v):
    """
    Rende un vettore verticale
    """
    return v.reshape((v.shape[0], 1))

def vrow(v):
    """
    Rende un vettore orizzontale
    """
    return v.reshape((1, v.shape[0]))

def compute_err_rate(prediction, labels):
    return ((prediction == labels).sum() / labels.size)

def calibration_set(scores, labels, partition=0.2):
    random.seed(42)
    limit = int(partition * scores.shape[0])
    
    evaluation_set, calibration_set  = scores[:limit], scores[limit:]
    evalutaion_label, calibration_label = labels[:limit], labels[limit:]
    return calibration_set, evaluation_set, calibration_label, evalutaion_label 
    

def k_folds(folds, labels, k, method, dim_reduction=None, m=None, preprocessing:str=None, SVM=False, **params):
    scores = []
    # iterate over 0 ... k
    for i in range(k):
        index_folds = [x for x in range(k) if x != i]
        # folds not equal to i (training)
        DTR = np.hstack([folds[i] for i in index_folds])
        # evaluation fold (the one equal to the loop index)
        DVAL = np.array(folds[i])
        
        if preprocessing:
            if preprocessing == 'z-norm':
                # print("Z-Normalization ----- enabled")
                DTR, DVAL = prep.preprocess_Z_score(DTR, DVAL)
            elif preprocessing == 'gau':
                # print("Gaussianization ----- enabled")
                if os.path.isfile(f"./data/gauss_folds/dtr-gauss-{i}-fold.npy") and os.path.isfile(f"./data/gauss_folds/dte-gauss-{i}-fold.npy"):
                    DTR = np.load(f"./data/gauss_folds/dtr-gauss-{i}-fold.npy")
                    DVAL = np.load(f"./data/gauss_folds/dte-gauss-{i}-fold.npy")
                else:
                    # print(f"Gauss fold [{i}]")
                    DTR, DVAL = prep.preprocess_gaussianization(DTR, DVAL)
                    np.save(f"./data/gauss_folds/dtr-gauss-{i}-fold.npy", DTR)
                    np.save(f"./data/gauss_folds/dte-gauss-{i}-fold.npy", DVAL)
                    # print("END")
        if dim_reduction:
            if dim_reduction == 'PCA':
                print("WARNING: PCA enabled ---- dimensionality reduction m:", m)
                P = prep.PCA(DTR, m)
                DTR = np.dot(P.T, DTR)
                DVAL = np.dot(P.T, DVAL)
            elif dim_reduction == 'LDA':
                pass
        # respective labels
        LTR = np.hstack([labels[i] for i in index_folds])
        LTE = np.array(labels[i])
        #print("\t\t[ Fold",i,"]")
        s = method(DTR, DVAL, LTR, params)
        if SVM:
            scores.append(s.ravel())
        else:
            scores.append(s)
        #prediction = s > 0
        #fold_acc = compute_err_rate(prediction, LTE)
        #print("\tAccuracy for fold[", i, "]:", fold_acc)
        #llr = SPost[1, :] / SPost[0, :]
    return scores

def pickle_dump(path, obj):
    with open(f'{path}.bin', 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    with open(f'{path}', 'rb') as handle:
        b = pickle.load(handle)
        return b
    return -1
    
def build_params(**params):
    return params

def shuffle(a: np.array, b: np.array, axis:int = 0):
    a_copy = deepcopy(a)
    b_copy = deepcopy(b)
    np.random.seed(42)
    p = np.random.permutation(a.shape[axis])
    if axis == 0:
        return a_copy[p], b_copy[p]
    if axis == 1:
        return a_copy[:,p], b_copy[:, p]
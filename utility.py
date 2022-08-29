import numpy as np

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
    return 1 - ((prediction == labels).sum() / labels.size)

def k_folds(folds, labels, k, method, classPriors, **params):
    scores = []
    # iterate over 0 ... k
    for i in range(k):
        index_folds = [x for x in range(3) if x != i]
        # folds not equal to i (training)
        DTR = np.hstack([folds[i] for i in index_folds])
        # evaluation fold (the one equal to the loop index)
        DVAL = np.array(folds[i])
        
        # respective labels
        LTR = np.hstack([labels[i] for i in index_folds])
        LTE = np.array(labels[i])
        print("\t\t[Fold",i,"]")
        s = method(DTR, DVAL, LTR, classPriors, params)
        scores.append(s)
        #prediction = s > 0
        #fold_acc = compute_err_rate(prediction, LTE)
        #print("\tAccuracy for fold[", i, "]:", fold_acc)
        #llr = SPost[1, :] / SPost[0, :]
    return scores
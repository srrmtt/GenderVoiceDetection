import numpy as np

N_SHAPES = 12


def load_dataset(filename):
    vett = []
    vettList = []
    labels = []

    f = open(filename)

    for line in f:
        try:
            row = line.split(",")
            vett = np.array([float(i) for i in row[0:N_SHAPES]]).reshape(N_SHAPES, 1)
            vettList.append(vett)
            labels.append(int(row[12]))
        except:
            print("Error reading file")
    f.close()
    return np.hstack(vettList), np.array(labels)

def z_norm(D):
    list_norm = []
    for i in range(N_SHAPES):
        column = D[i, :]
        mu = np.mean(column)
        sigma = np.cov(column)
        column_norm = (column - mu) / sigma
        list_norm.append(column_norm)
    return np.vstack(list_norm)

def make_folds(D, labels, k):
    # shuffle dataset
    shuffled_index = np.random.RandomState(seed=1234567).permutation(D.shape[1])
    D = D[:, shuffled_index]
    labels = labels[shuffled_index]
    # assign columns to the respective fold
    folds = []
    folds_label = []
    n = D.shape[1]
    
    fold_size = int(n / k)
    start = 0
    for i in range(k):
        if i == k - 1:
            limit = n
        else:
            limit = (i+1) * fold_size
        #print("taking split from", start, "to", limit ,D[:, start : limit])
        fold = D[:, start : limit]
        print("fold:", i," - ", fold.shape)
        folds.append(fold)
        folds_label.append(labels[start:limit])
        start = limit
    return folds, folds_label


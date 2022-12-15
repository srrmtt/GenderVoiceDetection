import numpy as np
import utility as util
import scipy.linalg

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

def preprocess_Z_score(DTR, DTE):
    print("Z-Scoring...")
    mu = util.vcol(DTR.mean(1))
    std = util.vcol(DTR.std(1))
    return (DTR - mu) / std, (DTE - mu) / std

def z_norm(D):
    mu = util.vcol(D.mean(1))
    std = util.vcol(D.std(1))
    return (D - mu) / std

def shuffle(D, L):
    shuffled_index = np.random.RandomState(seed=498540).permutation(D.shape[1])
    Dshaf = D[:, shuffled_index]
    Lshaf = L[shuffled_index]
    return Dshaf, Lshaf
def make_folds(D, labels, k):
    # shuffle dataset 1234567
    shuffled_index = np.random.RandomState(seed=498540).permutation(D.shape[1])
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

def compute_empirical_mean(X):
    return util.vcol(X.mean(1))

def compute_empirical_cov(X):
    mu = compute_empirical_mean(X)
    cov = np.dot((X - mu), (X - mu).T) / X.shape[1]
    return cov
def PCA(D, m):
    mu = D.mean(1)

    C = compute_empirical_cov(D)

    s, U = np.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]

    # project points
    DP = np.dot(P.T, D)
    return P

def compute_SwSb(D, L, classes):
    SB = 0
    SW = 0
    muG = compute_empirical_mean(D)

    for i in classes:
        Dc = D[:, L == i]
        mu = compute_empirical_mean(Dc)
        SB += D.shape[1] * np.dot((mu - muG), (mu - muG).T)
        SW += D.shape[1] * compute_empirical_cov(D[:, L == i])
    return SW/D.shape[1], SB / D.shape[1]

def LDA(D, L, classes, m = 1):
    
    SW, SB = compute_SwSb(D, L, classes)

    s, U = scipy.linalg.eigh(SB, SW)
    w = U[:, ::-1][:,0:m]

    
    return w
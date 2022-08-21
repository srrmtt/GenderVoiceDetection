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

import numpy as np
import scipy.special

import utility as util

def logpdf_GAU_ND(x, mu, C):
    P = np.linalg.inv(C)
    return -0.5 * x.shape[0] * np.log(np.pi * 2) + 0.5 * np.linalg.slogdet(P)[1] - 0.5 * (np.dot(P, (x-mu)) * (x-mu)).sum(0)

def ML_GAU(D):
    mu = util.vcol(D.mean(1))
    C = np.dot(D-mu, (D-mu).T) / float(D.shape[1])
    return mu, C

def computeTiedCovariance(D, L):
    C = np.zeros((D.shape[0],D.shape[0]))
    for lab in [0, 1]:
        Dlab = D[:, L == lab]
        C += ML_GAU(Dlab)[1] * float(Dlab.shape[1])
    return C / D.shape[1]

def MVG(DTR, DTE, LTR, params, classes=[0,1]):
    h = {}
    classPriors = params.setdefault('priors', [0.5, 0.5])
    diag = params.setdefault('diag', False)
    tied = params.setdefault('tied', False)

    for lab in classes:
        if tied:
            mu = util.vcol(DTR[:, LTR == lab].mean(1))
            C = computeTiedCovariance(DTR, LTR)
        else:
            mu, C = ML_GAU(DTR[:, LTR == lab])
        if diag:
            C = np.eye(C.shape[0]) * C
        h[lab] = (mu, C)
    
    logSJoint = np.zeros((2, DTE.shape[1]))

    for lab in classes:
        mu, C = h[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + np.log(classPriors[lab])
    
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    logPost = logSJoint - util.vrow(logSMarginal)
    post = np.exp(logPost)

    #make prediction with
    llr = np.log(post[1, :] / post[0, :])
  
   
    return llr
        
        



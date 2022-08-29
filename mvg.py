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


def MVG(DTR, DTE, LTR, classPriors, type_, classes=[0,1]):
    h = {}
    diag = type_.setdefault('diag', False)
    tied = type_.setdefault('tied', False)
    for lab in classes:
        if tied:
            mu, C = ML_GAU(DTR)
        else:
            mu, C = ML_GAU(DTR[:, LTR == lab])
        if diag:
            C = np.eye(C.shape[0]) * C
        h[lab] = (mu, C)
    
    logSJoint = np.zeros((3, DTE.shape[1]))

    for lab in classes:
        mu, C = h[lab]
        logSJoint[lab, :] = logpdf_GAU_ND(DTE, mu, C).ravel() + np.log(classPriors[lab])
    
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)

    logPost = logSJoint - util.vrow(logSMarginal)
    post = np.exp(logPost)

    #make prediction with
    llr = np.log(post[1, :] / post[0, :])
  
   
    return llr
        
        



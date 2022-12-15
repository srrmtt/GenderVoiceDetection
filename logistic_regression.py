import scipy.optimize
import numpy

import utility as util
import preprocessing as prep
def quadratic_trasformation(DTR: numpy.ndarray, DTE: numpy.ndarray):
    n_T = DTR.shape[1]
    n_E = DTE.shape[1]
    n_F = DTE.shape[0]
    n_F = n_F**2 + n_F
    quad_DTR = numpy.zeros((n_F, n_T))
    quad_DTE = numpy.zeros((n_F, n_E))
    for i in range(n_T):
        x = DTR[:, i:i+1]
        quad_DTR[:, i:i+1] = stack(x)
    for i in range(n_E):
        x = DTE[:, i:i+1]
        quad_DTE[:, i:i+1] = stack(x)
    return quad_DTR, quad_DTE


def stack(array):
    n_F = array.shape[0]
    xxT = array @ array.T
    column = numpy.zeros((n_F ** 2 + n_F, 1))
    for i in range(n_F):
        column[i*n_F:i*n_F + n_F, :] = xxT[:, i:i+1]
    column[n_F ** 2: n_F ** 2 + n_F, :] = array
    return column

def logreg_obj_wrap(D, labels, lam, priors):
    # Z = 1 id class = 1; -1 otherwise
    Z = labels * 2.0 - 1.0
    M = D.shape[0]
    

    def logreg_obj(v):
        w = util.vcol(v[0:M])
        b = v[-1]

        cxe = 0
         
        # use broadcasting
        for i in [0,1]:
            n = (labels == i).sum()
            S = numpy.dot(w.T, D[:,labels == i]) + b
            Z = i * 2.0 - 1.0
            cxe += numpy.logaddexp(0, -S * Z).sum() * priors[i] / n
        return cxe + lam * 0.5 * numpy.linalg.norm(w) ** 2

    return logreg_obj

def logreg(DTR, DTE, labels, params):
    priors = params['priors']
    lambda_ = params['lambda_']
    
    logreg_obj = logreg_obj_wrap(DTR, labels, lambda_, priors)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTE.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTE.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b

    
    return STE

def quadratic_logreg(DTR, DTE, LTE, params):
    DTR_quad, DTE_quad = quadratic_trasformation(DTR, DTE)
    
    return logreg(DTR_quad, DTE_quad, LTE, params)


def score_fusion(scoreA, scoreB, labels, priors):
    S = numpy.vstack((scoreA, scoreB))
    S, L = prep.shuffle(S, labels)
    
    dimDTR = int(0.7 * S.shape[1])
    lambda_ = 0
    DTR = S[:, 0:dimDTR]
    LTR = L[0:dimDTR]

    DTE = S[:, dimDTR:-1]
    LTE = L[dimDTR:-1]

    logreg_obj = logreg_obj_wrap(DTR, LTR, lambda_, priors)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTE.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTE.shape[0]]
    _b = _v[-1]

    fused_scores = _w[0] * DTE[0] + _w[1] * DTE[1] + _b
    print("end")
    return fused_scores, LTE
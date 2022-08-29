import scipy.optimize
import numpy

import utility as util

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

# usage

def logreg(DTR, DTE, labels, priors, params):
    lambda_ = params['lambda_']
    logreg_obj = logreg_obj_wrap(DTR, labels, lambda_, priors)
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTE.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTE.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b
    LP = STE > 0
    
    return STE
    
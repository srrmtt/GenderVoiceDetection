import scipy.optimize
import numpy
import dcf
import utility as util
import preprocessing as prep
from copy import deepcopy
from tqdm import tqdm

def logreg_obj_wrap(D, labels, lam):
    # Z = 1 id class = 1; -1 otherwise
    Z = labels * 2.0 - 1.0
    M = D.shape[0]
    
    def logreg_obj(v):
        w = util.vcol(v[0:M])
        b = v[-1]
        S = numpy.dot(w.T, D) + b
        cxe = numpy.logaddexp(0, -S * Z)
        return numpy.linalg.norm(w) ** 2 * lam / 2.0 + cxe.mean()

    return logreg_obj

def polynomial_trasformation(DTR: numpy.ndarray, DTE: numpy.ndarray):
    n_T = DTR.shape[1]
    n_E = DTE.shape[1]
    n_F = DTR.shape[0]
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


def weighted_logreg_obj_wrap(D, labels, lam, priors):
    # Z = 1 id class = 1; -1 otherwise
    Z = labels * 2.0 - 1.0
    M = D.shape[0]
    
    
    def logreg_obj_weighted(v):
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
    
   
    return logreg_obj_weighted
    

def logreg(DTR, DTE, labels, params):
    priors = params['priors']
    lambda_ = params['lambda_']
    weighted = params.get('weighted', False)
    score_cal = params.get('score_cal', False)
    
    logreg_obj = weighted_logreg_obj_wrap(DTR, labels, lambda_, priors) if weighted else logreg_obj_wrap(DTR, labels, lambda_)
    
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTE.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTE.shape[0]]
    _b = _v[-1]
    if score_cal:
        print(_w.shape)
        _b = _v[-1] - numpy.log(priors[0] / (1 - priors[0])) 
    STE = numpy.dot(_w.T, DTE) + _b

    
    return STE

def score_fusion(scoreA, scoreB, labels, priors, perc=0.7):
    S = numpy.vstack((scoreA, scoreB))
    L = numpy.vstack((labels, labels))
    
    print(S.shape, L.shape)
    
    # S, L = util.shuffle(S, L, axis=1)
    lambda_ = 0
    
    
    limit = int(perc * S.shape[1])
    
    DTR, DTE  = S[:, :limit], S[:, limit:]
    LTR, LTE = L[:, :limit], L[:, limit:]
    

    fused_scores = logreg(DTR, DTE, LTR[0], {'priors': [0.5,0.5], 'lambda_' : lambda_, 'weighted':True})

     
    return fused_scores, LTE, limit

def score_calibration(calibration_set, eval_set, calibration_labels, weighted: bool = True):
    prior = [0.5, 0.5]
    params = {
        'priors' : prior,
        'lambda_' : 0,
        'weighted' : True,
        'score_cal': True
    }
    return logreg(calibration_set, eval_set, calibration_labels, params) 
    
    
def quadratic_logreg(DTR:numpy.array, DTE:numpy.array, labels:numpy.array, params:dict):
    """
    Quadratic version of the logistic regression, it takes the training dataset and the relative labels and
    compute the w and b to project the DTE dataset passed as input. The params dictionary contains all the properties
    of the logreg as priors, lambda and weighted version.
    """
    priors = params['priors']
    lambda_ = params['lambda_']
    weighted = params.get('weighted', False)
    
    
    DTR, DTE = polynomial_trasformation(DTR, DTE)
    if weighted:
        logreg_obj = weighted_logreg_obj_wrap(DTR, labels, lambda_, priors)
    else:
        logreg_obj = logreg_obj_wrap(DTR, labels, lambda_)
    
    _v, _J, _d = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTE.shape[0] + 1), approx_grad=True)
    _w = _v[0:DTE.shape[0]]
    _b = _v[-1]
    STE = numpy.dot(_w.T, DTE) + _b
    
    return STE


def minDCF_from_scores(scores: dict, labels: numpy.array) -> dict:
    """
    From a list of scores return the minDCFs for the applications in a dictionary format.
    """
    ret = dict()
    for application, piT_scores in scores.items():
        minDCFs = dict()
        pi, Cfn, Cfp = application
        for pi_t, score in piT_scores.items():
            minDCFs[pi_t] = dcf.compute_min_DCF(score, labels, pi, Cfn, Cfp)
        ret[application] = deepcopy(minDCFs)
    return ret

def load_results(prefix: str) -> tuple:
    TRAIN_PATH = "./results/logreg/plots/"
    
    min_dcf_filename = f"{prefix}_min_dcfs.bin"
    lambdas_filename = f"{prefix}_lambdas.npy"
    
    minDCFs = util.pickle_load(f"{TRAIN_PATH}/{min_dcf_filename}")
    lambdas = numpy.load(f"{TRAIN_PATH}/{lambdas_filename}")
    
    return minDCFs, lambdas

def compute_filename_prefix(quadratic: bool, preprocessing: str, weighted: bool) -> str:
    """
    util method that gives the filename from a set of logreg option. 
    """
    quadratic_ = "quadratic" if quadratic else "linear" 
    raw_ = "raw" if not preprocessing else preprocessing
    weighted_ = "weighted" if weighted else "not-weighted" 
    
    prefix = f"{weighted_}-{raw_}-plot_{quadratic_}"
    return prefix

def compute_minDCF_for_lambda(DTR, DTE, LTR, LTE, application: tuple, lambdas: list, quadratic: bool, params: dict):
    """
    This function is used in the plots functions in order to compute the minDCF for a set of lambdas using quadratic 
    or linear logistic regression.
    """
    DCFs = []
    for l in tqdm(lambdas):
        params['lambda_'] = l
        if not quadratic:
            scores = logreg(DTR, DTE, LTR, params)
        else:
            scores = quadratic_logreg(DTR, DTE, LTR, params)
        
        pi, Cfn, Cfp = application
        DCF = dcf.compute_min_DCF(scores, LTE, pi, Cfn, Cfp) 
        DCFs.append(DCF)
    return DCFs
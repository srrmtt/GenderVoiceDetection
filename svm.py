import numpy
import scipy
import utility
from tqdm import tqdm
import utility as util
import dcf
from copy import deepcopy 
def minDCF_from_scores(scores: dict, labels: numpy.array) -> dict:
    ret = dict()
    for application, piT_scores in scores.items():
        minDCFs = dict()
        pi, Cfn, Cfp = application
        for pi_t, score in piT_scores.items():
            minDCFs[pi_t] = dcf.compute_min_DCF(score, labels, pi, Cfn, Cfp)
        ret[application] = deepcopy(minDCFs)
    return ret

def compute_weight_C(C, LTR, prior):
    bounds = numpy.zeros((LTR.shape[0]))
    pi_t_emp = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * prior[1] / pi_t_emp
    bounds[LTR == 0] = C * prior[0] / (1 - pi_t_emp)
    return list(zip(numpy.zeros(LTR.shape[0]), bounds))

def train_SVM_linear(DTR, DTE, LTR, params, K = 0):
    C = params.setdefault('C', 0.1)
    balanced_classes = params.setdefault('balanced', False)
    priors = params.setdefault('priors', [0.5, 0.5])

    DTRext = numpy.vstack([DTR, numpy.ones( (1, DTR.shape[1]) ) ])

    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.dot(DTRext.T, DTRext)
    H = utility.vcol(Z) * utility.vrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, utility.vcol(alpha))
        aHa = numpy.dot(utility.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad
    
    def JPrimal(w):
        S = numpy.dot(utility.vrow(w), DTRext)
        loss = numpy.maximum(numpy.zeros(S.shape), 1 - Z * S).sum()
        return 0.5 * numpy.linalg.norm(w) ** 2 + C * loss
    
    if balanced_classes:
        bounds = compute_weight_C(C, LTR, priors)
    else:
        bounds = [(0,C)] * DTR.shape[1]
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds= bounds,
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )
    
    wstar = numpy.dot(DTRext, utility.vcol(alphaStar) * utility.vcol(Z))

    #print("\t\tDuality Gap:", numpy.abs(JDual(alphaStar)[0] - JPrimal(wstar))) # the difference must be very low
    # compute scores 
    DTEext = numpy.vstack([DTE, numpy.ones( (1, DTE.shape[1]) ) ])
    scores = numpy.dot(wstar.T, DTEext)

    return scores

# ---------------------- NON LINEAR SVM -------------------------


# --- RBF kernel ---
def RBF_kernel(DTR, gamma, K):
    dist = utility.vcol((DTR ** 2).sum(0)) + utility.vrow((DTR ** 2).sum(0)) - 2 * numpy.dot(DTR.T, DTR)
    H = numpy.exp(-gamma * dist) + K
    return H

def RBF_sample(x1, x2, gamma):
    return numpy.exp(-gamma * numpy.linalg.norm(x1 - x2)**2)

def RBF_score(xt, DTR, alpha, Z, gamma):
    score = 0
    for i in range(DTR.shape[1]):
        if alpha[i] > 0:
            score += alpha[i] * Z[i] * RBF_sample(DTR[:, i], xt, gamma)
    return score

def rbf_score_matrix(DTR, DTE, Z, alpha_opt, K, gamma):
    exp_dist = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            exp_dist[i][j] += numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i:i+1] - DTE[:, j:j+1]) ** 2) + (K)**2
    return (util.vcol(alpha_opt) * util.vcol(Z) * (exp_dist)).sum(0)
# --- poly kernel ---

def poly_sample(x1, x2, c, degree=2):  
    return (numpy.dot(x1.T, x2) + c)**degree

def poly_kernel(DTR, Z ,c, degree, K):
    H = (numpy.dot(DTR.T, DTR) + c) ** degree + K ** 2

    return H

def poly_score(xt, DTR, alpha, Z, d, c):
    score = 0
    for i in range(DTR.shape[1]):
        if alpha[i] > 0:
            score += alpha[i] * Z[i] * poly_sample(DTR[:, i], xt, c, d)
    return score




def train_non_linear_SVM(DTR, DTE, LTR, params):
    C = params.setdefault('C', 1)
    kernel = params.setdefault('kernel', 'rbf')
    K = params.setdefault('K', 1)
    balanced_classes = params.setdefault('balanced', False)
    priors = params.setdefault('priors', [0.5, 0.5])

    if kernel == 'rbf':
        gamma = params.setdefault('gamma', 1.0)
    
    if kernel == 'poly':
        d = params.setdefault('d', 2)
        c = params.setdefault('c', 0)
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.array([])
    if(kernel == 'rbf'):
        H = RBF_kernel(DTR, gamma, K)
    if(kernel == 'poly'):
        H = poly_kernel(DTR, Z, c, d, K)

    H = utility.vcol(Z) * utility.vrow(Z) * H

    def JDual(alpha):
        Ha = numpy.dot(H, utility.vcol(alpha))
        aHa = numpy.dot(utility.vrow(alpha), Ha)
        a1 = alpha.sum()
        return -0.5 * aHa.ravel() + a1, -Ha.ravel() + numpy.ones(alpha.size)
    
    def LDual(alpha):
        loss, grad = JDual(alpha)
        return -loss, -grad

    if balanced_classes:
        pi_T_emp = (LTR == 1).sum() / LTR.size
        pi_F_emp = 1 - pi_T_emp
        C_T = C * (priors[0] / pi_T_emp)
        C_F = C*(priors[1] / pi_F_emp)

        bounds = [(0, C_T) if lab == 1 else (0, C_F) for lab in LTR ]
    else:
        bounds = [(0,C)] * DTR.shape[1]
    
    alphaStar, _x, _y = scipy.optimize.fmin_l_bfgs_b(
        LDual,
        numpy.zeros(DTR.shape[1]),
        bounds=bounds,
        factr=100.0,
        maxiter=100000,
        maxfun=100000,
    )
    
    scores = []

    if kernel == 'rbf':
        return rbf_score_matrix(DTR, DTE, Z, alphaStar, K, gamma)
    if kernel == 'poly':
        scores = (util.vcol(alphaStar) * util.vcol(Z) * ((numpy.dot(DTR.T, DTE) + c) ** d + K**2)).sum(0)
        return scores
    for t in range(DTE.shape[1]):
        scores.append(score)
    scores = numpy.array(scores)
    
    return util.vrow(scores)

def compute_filename_prefix(balanced: bool, preprocessing: bool) -> str:
    balanced_ = "balanced" if balanced else "not-balanced" 
    preprocessing_ = "z-norm" if preprocessing else "raw"
    return f"{balanced_}-{preprocessing_}"

def load_results(filename_prefix, kernel: str):
    Cs_PATH = f"./results/svm/plots/plot_{kernel}_Cs.npy"
    MINDCF_PATH = f"./results/svm/plots/{filename_prefix}-plot_{kernel}_minDCFs.bin"
    minDCFs = util.pickle_load(MINDCF_PATH)
    Cs = numpy.load(Cs_PATH)
    
    return minDCFs, Cs

def compute_minDCF_for_parameter(DTR, DTE, LTR, LTE, evaluation_point: tuple, parameters: list, params: dict):
    kernel = params.get('kernel', 'linear')
    DCFs = []
    for hyperparam in tqdm(parameters):
        params['C'] = hyperparam
        if kernel == 'linear':
            pi, Cfn, Cfp = evaluation_point
            scores = train_SVM_linear(DTR, DTE, LTR, params)
        elif kernel == 'poly':
            pi, Cfn, Cfp = evaluation_point            
            scores = train_non_linear_SVM(DTR, DTE, LTR, params)
        elif kernel == 'rbf':
            pi, Cfn, Cfp = (0.5, 1, 1)
            scores = train_non_linear_SVM(DTR, DTE, LTR, params)
        else:
            print(kernel)
            print("Specify a kernel from [linear, poly, rbf]")
            return -1
        
        minDCF = dcf.compute_min_DCF(scores.ravel(), LTE, pi, Cfn, Cfp)
        DCFs.append(minDCF)
    return DCFs
        
    
import numpy
import scipy
import utility

import utility as util
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
        bounds= bounds,
        factr=1.0,
        maxiter=100000,
        maxfun=100000,
    )
    
    wstar = numpy.dot(DTRext, utility.vcol(alphaStar) * utility.vcol(Z))

    print("\t\tDuality Gap:", numpy.abs(JDual(alphaStar)[0] - JPrimal(wstar))) # the difference must be very low
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

def compute_RBFSVM_score_matrix(DTR, DTE, Z, alpha_opt, K, gamma):
    exp_dist = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            exp_dist[i][j] += numpy.exp(-gamma * numpy.linalg.norm(DTR[:, i:i+1] - DTE[:, j:j+1]) ** 2) + (K)**2
    return (util.vcol(alpha_opt) * util.vcol(Z) * (exp_dist)).sum(0)
# --- poly kernel ---

def poly_sample(x1, x2, c, d):  
    return (numpy.dot(x1.T, x2) + c)**d

def poly_kernel(DTR, c, d, K):
    H = []
    for i in range(DTR.shape[1]):
        Hi = []
        for j in range(DTR.shape[1]):
            Hi.append(poly_sample(DTR[:,i],DTR[:,j],c,d)+K)
        H.append(Hi)
    return numpy.array(H)

def poly_score(xt, DTR, alpha, Z, d, c):
    score = 0
    for i in range(DTR.shape[1]):
        if alpha[i] > 0:
            score += alpha[i] * Z[i] * poly_sample(DTR[:, i], xt, c, d)
    return score




def train_non_linear_SVM(DTR, DTE, LTR, params, balanced=False):
    C = params.setdefault('C', 1)
    kernel = params.setdefault('kernel', 'rbf')
    K = params.setdefault('K', 1)
    balanced_classes = params.setdefault('balanced', False)
    priors = params.setdefault('priors', [0.5, 0.5])

    if kernel == 'rbf':
        gamma = params.setdefault('gamma', 1.0)
    
    if kernel == 'poly':
        d = params.setdefault('d', 2)
        c = params.setdefault('c', 1)
    
    Z = numpy.zeros(LTR.shape)
    Z[LTR == 1] = 1
    Z[LTR == 0] = -1

    H = numpy.array([])
    if(kernel == 'rbf'):
        H = RBF_kernel(DTR, gamma, K)
    if(kernel == 'poly'):
        H = poly_kernel(DTR, c, d, K)

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
    
    #wstar = numpy.dot(DTRext, utility.vcol(alphaStar) * utility.vcol(Z))

    print(JDual(alphaStar)[0]) 
    scores = []

    if kernel == 'rbf':
        return compute_RBFSVM_score_matrix(DTR, DTE, Z, alphaStar, K, gamma)
        
    for t in range(DTE.shape[1]):
        if kernel == 'poly':
            score = poly_score(DTE[:, t], DTR, alphaStar, Z, d, c)
        scores.append(score)
    scores = numpy.array(scores)
    
    return util.vrow(scores)
# for a test sample
# sum_over_i alpha_i * z_i * K(x_i, x_t)


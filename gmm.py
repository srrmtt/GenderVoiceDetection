import numpy 
import scipy


import utility as util
import dcf

def logpdf_GAU_ND_Opt(X, mu, C):
    # function for the log likelihood 
    #print("svd:", numpy.linalg.svd(C)[1])
    P = numpy.linalg.inv(C)
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi)
    const += -0.5 * numpy.linalg.slogdet(C)[1]


    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        res = const + -0.5 * numpy.dot( (x-mu).T, numpy.dot(P, (x-mu)))
        Y.append(res)

    return numpy.array(Y).ravel()

def GMM_ll_perSample(X, gmm):
    # compute log likelihood for each samples, so return an array of ll for each of the samples contained in X
    G = len(gmm)
    N = X.shape[1]
    S = numpy.zeros((G, N))
    for g in range(G):
        # for each component we compute the likelihood of the corrisponding gaussian for the samples. We add the value of the weight, that is the prior 
        # of the component. This for compute the Joint matrix 
        S[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
       
    # take the sum of the exponential of the row of matrix Joint

    return scipy.special.logsumexp(S, axis=0)
def mrow(v):
    return v.reshape((1, v.shape[0]))
def mcol(v):
    return v.reshape((v.shape[0], 1))


def GMM_tied(GMM, Z_vec, N):

    tied_Sigma = numpy.zeros((GMM[0][2].shape[0], GMM[0][2].shape[0]))
    for g in range((len(GMM))):
        tied_Sigma += GMM[g][2] * Z_vec[g]
    tied_Sigma = (1/N) * tied_Sigma
    for g in range((len(GMM))):
        GMM[g] = (GMM[g][0], GMM[g][1], tied_Sigma)
    return GMM

def GMM_EM(X, gmm, full_cov=True, threshold=1e-6, psi=0.01,tied=False):
    # ll of the current gmm
    print("Model full_cov", full_cov, "--- Tied:", tied)
    llNew = None
    llOld = None
    lastll = 0
    G = len(gmm)
    N = X.shape[1]
    
    #if tied:
    #    return _GMM_EM_tied(X, gmm)
    # this difference represents the difference between the two likelihoods
    while llOld is None or llNew - llOld > threshold:
        llOld = llNew
        SJ = numpy.zeros((G, N))
        ##### compute the matrix of Joint density like in the previous function####
        for g in range(G):
            SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
        # marginal probability (like MVG classifier)
        SM = scipy.special.logsumexp(SJ, axis=0)
        # log likelihood for the all dataset (indipendent samples hypothesis)
        llNew = SM.sum() / N
        # component posterior probabilities for each sample in log domain (-)
        P = numpy.exp(SJ - SM)
        # generate the updated parameters of our GMM
        gmmNew = []
        sigma_sum = numpy.zeros((X.shape[0], X.shape[0]))
        # compute min covariance matrix and weight
        for g in range(G):
            gamma = P[g, :]
            # zero order statistic
            Z = gamma.sum()
            # first order and second order statistics
            # evaluate F with broadcasting
            F = (mrow(gamma) * X).sum(1)
            S = numpy.dot(X ,(mrow(gamma) * X).T)
            # weight
            w = Z / N
            # mean
            mu = mcol(F / Z)
            # covariance matrix
            Sigma = S/Z - numpy.dot(mu, mu.T)
            #print('Sigma: ',Sigma)
            if not full_cov:
                Sigma = Sigma * numpy.eye(Sigma.shape[0])
            if tied:
                sigma_sum += Z*Sigma
            
            else:
                U, s, _ = numpy.linalg.svd(Sigma)
                s[s < psi] = psi
                Sigma = numpy.dot(U, mcol(s)*U.T)
                
                    

            gmmNew.append((w, mu, Sigma))
        if tied:
            Sigma = sigma_sum / G
            U, s, _ = numpy.linalg.svd(Sigma)
            s[s < psi] = psi
            Sigma = numpy.dot(U, mcol(s)*U.T)
        
            
                
           
        
        gmm = gmmNew
        # check if the llNew in increasing used for bugs
        if(llNew < lastll and lastll != 0):
            #print("ERROR decreasing ll")
            pass
        lastll = llNew
        #print("new ll:", llNew)
    
    #print("diff:", llNew - llOld)
    return gmm
def _GMM_EM_tied(X, gmm):
        print("call the new function")
        ll_new = None
        ll_old = None
        G = len(gmm)
        N = X.shape[1]
        
        psi = 0.01
        
        while ll_old is None or ll_new-ll_old>1e-6:
            ll_old = ll_new
            SJ = numpy.zeros((G, N))
            for g in range(G):
                SJ[g, :] = logpdf_GAU_ND_Opt(X, gmm[g][1], gmm[g][2]) + numpy.log(gmm[g][0])
            SM = scipy.special.logsumexp(SJ, axis=0)
            ll_new = SM.sum() / N
            P = numpy.exp(SJ - SM)
            
            gmm_new = []
            summatory = numpy.zeros((X.shape[0], X.shape[0]))
            for g in range(G):
                gamma = P[g, :]
                Z = gamma.sum()
                F = (mrow(gamma)*X).sum(1)
                S = numpy.dot(X, (mrow(gamma)*X).T)
                w = Z/N
                mu = mcol(F/Z)
                sigma = S/Z - numpy.dot(mu, mu.T)
                summatory += Z*sigma
                gmm_new.append((w, mu, sigma))
            #tying
            sigma = summatory / G
            #constraint
            U, s, _ = numpy.linalg.svd(sigma)
            s[s<psi] = psi
            sigma = numpy.dot(U, mcol(s)*U.T)
            gmm = gmm_new
            #print(ll_new)
        #print(ll_new-ll_old)
        return gmm
def empirical_cov(D):
    mu = D.mean(1)
    #print(mu,mu.shape)
    DC = D - mu.reshape(mu.size,1)
    return numpy.dot(DC,DC.T) / float(DC.shape[1])

def split(GMM, alpha):
    U, s, Vh = numpy.linalg.svd(GMM[2])
    d = U[:, 0:1] * s[0]**0.5 * alpha
    return (GMM[0] / 2, GMM[1] + d, GMM[2]), (GMM[0] / 2, GMM[1] - d, GMM[2])

def LBG(gmms, alpha):
    ret = []

    for gmm in gmms:
        C = constrEigenvaluesCovMat(gmm[2], 0.01)
        gmmConstr = (gmm[0], gmm[1], C)
        gmm1, gmm2 = split(gmmConstr, alpha)
        ret.append(gmm1)
        ret.append(gmm2)
    
    return ret
    

# Constraining the eigenvalues of the covariance matrices
def constrEigenvaluesCovMat(C, psi):
    U, s, _ = numpy.linalg.svd(C)
    #print("s:", s)
    s[s<psi] = psi
    new_C = numpy.dot(U, mcol(s) * U.T)
    
    return new_C


def diag_cov(DTR):
    C = empirical_cov(DTR)
    return C * numpy.eye(C.shape[0])
# compute_cov, alpha, stopping_criterion, G, psi, cov_type='full',tied=False
def GMM(DTR, DTE, LTR, params={}):
    priors = params.setdefault('priors', [0.5, 0.5])
    alpha = params.setdefault('alpha', 0.1)
    stopping_criterion = params.setdefault('stopping_criterion', 10**-6)
    G = params.get('G')
    psi = params.setdefault('psi', 0.01)
    full = params.setdefault('full_cov', True)
    tied = params.setdefault('tied', False)

    if full:
        compute_cov = empirical_cov
    else:
        compute_cov = diag_cov

    lls = []
    for i in range(2):
        ll_for_class = []
        class_data = DTR[:, LTR == i]
        Sigma = constrEigenvaluesCovMat(compute_cov(class_data), psi)
        start_gmm = [(1, mcol(numpy.mean(class_data, 1 )), Sigma)]
        
        #print("\t\tprocessing class:", i)
        lbg_gmm = start_gmm
        for j in range(G):
            #print("\t\twith component(s):", len(lbg_gmm))
            new_gmm = GMM_EM(class_data, lbg_gmm, full, stopping_criterion, psi,tied)
            start_gmm = new_gmm
            ll = GMM_ll_perSample(DTE, new_gmm)
            ll_for_class.append(ll)
            lbg_gmm = LBG(start_gmm, alpha)
            
            for i,gmm in enumerate(lbg_gmm):
                Cconstr = constrEigenvaluesCovMat(gmm[2], psi)
                lbg_gmm[i] = (gmm[0], gmm[1], Cconstr)
        lls.append(ll_for_class)
    llrs = []
    
    for i in range(G):  
        llr = lls[1][i] - lls[0][i]    
        llrs.append(llr)
    llrs = numpy.vstack(llrs)
    print("llrs shape", llrs.shape)
    return llrs


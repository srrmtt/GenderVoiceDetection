import numpy as np
import scipy as sp
from matplotlib.pyplot import cm
# my libraries import 
import preprocessing as prep
import plot as plt
import mvg
import utility as util
import logistic_regression as lr
import dcf 
import gmm 

N_FEATURES = 12

DATA_VISUALIZATION = False
MVG = False
LOGISTIC_REGRESSION = False
GMM = True

if __name__ == '__main__':
    fileTR = './data/Train.txt'
    fileTE = './data/Train.txt'

    features = ["Feature(" + str(x) + ")" for x in range(N_FEATURES)]

    DTR, LTR = prep.load_dataset(fileTR)
    DTE, LTE = prep.load_dataset(fileTE)
    print(DTR.shape)
    #plt.plot_features_distr(DTR, LTR, features)
    
    DTR = prep.z_norm(DTR)
    DTE = prep.z_norm(DTE)
    classPriors = [1/2, 1/2]

    ### DATA VISUALIZATION ###
    if DATA_VISUALIZATION:
        maleDTR = DTR[:, LTR == 0]
        femaleDTR = DTR[:, LTR == 1]
        plt.plot_features_distr(DTR, LTR, features)
        plt.plot_relation_beetween_feautures(DTR, LTR, features)
        plt.plot_heatmap(DTR, features, cm.Greys)
        plt.plot_heatmap(maleDTR, features, cm.Blues)
        plt.plot_heatmap(femaleDTR, features, cm.Reds)

    applications = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]
    k = 3
    folds, folds_labels = prep.make_folds(DTR, LTR, k)

    ### MVG ###
    if MVG:
        print("----- MVG full covariance -----")
        for application in applications:
            pi = application[0]
            Cfn = application[1]
            Cfp = application[2]
            print("Application with (pi:", pi,", Cfn",Cfn,", Cfp",Cfp,")")
            classPriors = [pi, 1-pi]
            llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, classPriors)
            scores = np.hstack(llrs)
        
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
            print("minDCF:", minDCF)

    ### LOGISTIC REGRESSION ###
    if LOGISTIC_REGRESSION:
        print("\n\n----- Linear logistic regression -----")
        #plt.plot_min_DCF(folds, folds_labels, k, applications)
        for application in applications:
            pi = application[0]
            Cfn = application[1]
            Cfp = application[2]
            
            print("Application with ( pi:", pi,", Cfn:",Cfn,", Cfp:",Cfp,")")
            classPriors = [pi, 1-pi]
            STE = util.k_folds(folds, folds_labels, k, lr.logreg, classPriors, lambda_ = 10**-6)
            scores = np.hstack(STE)
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
            print("minDCF:", minDCF)

    
        
    ### GMM Models ###
    if GMM:
        print("\n\n----- GMM Classifier -----")
        print("\tFull Covariance - Non Tied Cvoariances")
        alpha = 0.1 
        stopping_criterion = 10**-6 
        G = 3 
        psi = 0.01 
        full_cov = True 
        tied = False

        folds_component_llrs = util.k_folds(folds, folds_labels, k, gmm.GMM, classPriors,
                            alpha=alpha, stopping_criterion=stopping_criterion, G=G, psi=psi, full_cov=full_cov, tied=tied )
        minDCFs = dcf.GMM_minDCF(folds_component_llrs, folds_labels, G, k, applications[0])

        plt.plot_minDCF_GMM_hist(minDCFs, G)

        # print("\tFull Covariance - Tied Cvoariances")
        # folds_component_lls = util.k_folds(folds, folds_labels, k, gmm.GMM, classPriors,
        #                     alpha=alpha, stopping_criterion=stopping_criterion, G=G, psi=psi, full_cov=full_cov, tied=True)
        # minDCFs = dcf.GMM_minDCF(folds_component_lls, folds_labels, G, k, applications[0])

        # print("\tDiag Covariance - Non Tied Cvoariances")
        # llrs = util.k_folds(folds, folds_labels, k, gmm.GMM, classPriors,
        #                     alpha=alpha, stopping_criterion=stopping_criterion, G=G, psi=psi, full_cov=False, tied=False)
        # minDCFs = dcf.GMM_minDCF(folds_component_lls, folds_labels, G, k, applications[0])
        
        # print("\tDiag Covariance - Tied Cvoariances")
        # llrs = util.k_folds(folds, folds_labels, k, gmm.GMM, classPriors,
        #                     alpha=alpha, stopping_criterion=stopping_criterion, G=G, psi=psi, full_cov=True, tied=True)
        # minDCFs = dcf.GMM_minDCF(folds_component_lls, folds_labels, G, k, applications[0])
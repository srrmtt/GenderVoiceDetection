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
import svm

N_FEATURES = 12

DATA_VISUALIZATION = False
MVG = False
LOGISTIC_REGRESSION = False
SVM = False
GMM = True
DIM_REDUCTION = False
PREPROCESSING = False
FUSION = False

if __name__ == '__main__':
    fileTR = './data/Train.txt'
    fileTE = './data/Train.txt'

    features = ["Feature(" + str(x) + ")" for x in range(N_FEATURES)]

    DTR, LTR = prep.load_dataset(fileTR)
    DTE, LTE = prep.load_dataset(fileTE)
        


    classPriors = [1/2, 1/2]
    m = None
    PCA_enabled = False

    
    if DIM_REDUCTION:
        #print("---- LDA ----")
        #DTR_lda = prep.LDA(DTR, LTR, [0,1])
        #print("space reducted to:", DTR_lda.shape)

        m = 9
        PCA_enabled = True
        print("---- PCA with m=", m," -----")
        

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
    k = 4
    foldsZ, folds_labels = prep.make_folds(DTR, LTR, k)
    folds, folds_labels = prep.make_folds(DTR, LTR, k)

    ### MVG ###
    if MVG:
        # pi = 0.5
        # classPriors = [pi, 1-pi]
        # llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, PCA_enabled=PCA_enabled, m=m, classPriors=classPriors)
        # scoresMVG = np.hstack(llrs)
        print("----- MVG full covariance -----")
        for application in applications:
            pi, Cfn, Cfp = application
            print("Application with (pi:", pi,", Cfn",Cfn,", Cfp",Cfp,")")
            classPriors = [pi, 1-pi]
            llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, PCA_enabled=PCA_enabled, m=m, classPriors=classPriors)
            scores = np.hstack(llrs)
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
            #print("minDCF:", minDCF)
        print("----- MVG diagonal covariance -----")
        for application in applications:
            pi, Cfn, Cfp = application
            print("Application with (pi:", pi,", Cfn",Cfn,", Cfp",Cfp,")")
            classPriors = [pi, 1-pi]
            llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, PCA_enabled=PCA_enabled, m=m, classPriors=classPriors, diag=True)
            scores = np.hstack(llrs)
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        
        print("----- MVG tied full covariance -----")
        for application in applications:
            pi, Cfn, Cfp = application
            print("Application with (pi:", pi,", Cfn",Cfn,", Cfp",Cfp,")")
            classPriors = [pi, 1-pi]
            llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, PCA_enabled=PCA_enabled, m=m, classPriors=classPriors, tied=True)
            scores = np.hstack(llrs)
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        
        print("----- MVG tied digonal covariance -----")
        for application in applications:
            pi, Cfn, Cfp = application
            print("Application with (pi:", pi,", Cfn",Cfn,", Cfp",Cfp,")")
            classPriors = [pi, 1-pi]
            llrs = util.k_folds(folds, folds_labels, k, mvg.MVG, PCA_enabled=PCA_enabled, m=m, classPriors=classPriors, diag=True, tied=True)
            scores = np.hstack(llrs)
            minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)

    ### LOGISTIC REGRESSION ###
    if LOGISTIC_REGRESSION:
        print("\n\n----- Linear logistic regression -----")
        classPriors = [0.5, 0.5]
        STE = util.k_folds(folds, folds_labels, k, lr.logreg, priors=classPriors, lambda_ = 10**-6)
        scoresLR = np.hstack(STE)
        #plt.plot_min_DCF_logreg(folds, folds_labels, k, applications, quadratic=False)
        # for application in applications:
        #     pi = application[0]
        #     Cfn = application[1]
        #     Cfp = application[2]
            
        #     l = 10e-6
        #     print("Application with ( pi:", pi,", Cfn:",Cfn,", Cfp:",Cfp,")")
        #     for pi_T in [0.5, 0.1, 0.9]:
        #         print("\tevaluating with pi_T:", pi_T)
        #         classPriors = [pi_T, 1-pi_T]
        #         STE = util.k_folds(folds, folds_labels, k, lr.logreg, priors=classPriors, lambda_ = 10**-6)
        #         scores = np.hstack(STE)
        #         minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        #         #print("minDCF:", minDCF)

        # print("------ Quadratic Logistic Regression ------")
        # #plt.plot_min_DCF_logreg(folds, folds_labels, k, applications, quadratic=True)
        # for application in [applications[2]]:
        #     pi = application[0]
        #     Cfn = application[1]
        #     Cfp = application[2]
            
        #     print("Application with ( pi:", pi,", Cfn:",Cfn,", Cfp:",Cfp,")")
        #     for pi_T in [0.5, 0.1, 0.9]:
        #         print("\tevaluating with pi_T:", pi_T)
        #         classPriors = [pi_T, 1-pi_T]
        #         STE = util.k_folds(folds, folds_labels, k, lr.quadratic_logreg, priors=classPriors, lambda_ = 10**-6)
        #         scores = np.hstack(STE)
        #         minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)

    
    ### SVM ###
    if SVM:
        # print("\n\n----- linear SVM -----")
        #plt.plot_min_DCF_svm(folds, folds_labels, k, applications, balanced=True)
        pi, Cfn, Cfp = applications[0]
        # scoresSVM = util.k_folds(folds, folds_labels, k, svm.train_SVM_linear, C = 1)
        # scores = np.hstack(scores)
        # for application in applications:
        #     pi, Cfn, Cfp = application
        #     print("application:", application)
        #     minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        # scores = util.k_folds(folds, folds_labels, k, svm.train_SVM_linear, C = 1, balanced=True)
        # scores = np.hstack(scores)
        # for application in applications:
        #     pi, Cfn, Cfp = application
        #     print("application:", application)
        #     minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        
        print("----- Exponential SVM ----")
        #plt.plot_min_DCF_svm(folds, folds_labels, k, applications)
        
        for application in applications:
            pi, Cfn, Cfp = application
            print("application:", application)
            for pi_T in [0.5, 0.1, 0.9]:
                print("\tevaluating with pi_T:", pi_T)
                classPriors = [pi_T, 1-pi_T]
                scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM, kernel='rbf', preprocessing=False, C=0.10, gamma=0.01, balanced=True, priors=classPriors)
                scores = np.hstack(scores)
                  
                minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)
        

        # print("----- Quadratic SVM ----")
        # #plt.plot_min_DCF_svm(folds, folds_labels, k, applications)
        # scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM, kernel='poly', C=0.1, d=2.0, c=1)
        # scores = np.hstack(scores)
        # for application in applications:
        #     pi, Cfn, Cfp = application
        #     print("application:", application)
        #     minDCF = dcf.compute_min_DCF(scores, np.hstack(folds_labels), pi, Cfn, Cfp)


    ### GMM Models ###
    if GMM:
        print("\n\n----- GMM Classifier -----")
        print("\tFull Covariance - Non Tied Cvoariances")
        alpha = 0.1 
        stopping_criterion = 1e-6
        G = 2
        psi = 0.01 
 

        folds_component_llrs = util.k_folds(folds, folds_labels, k, gmm.GMM, PCA_enabled, None, preprocessing=True,
                            alpha=alpha, stopping_criterion=stopping_criterion, G=G, psi=psi, full_cov=False, tied=False )
        minDCFsRaw = dcf.GMM_minDCF(folds_component_llrs, folds_labels, G, k, applications[0])

        

        
    if FUSION:
        scores, labels = lr.score_fusion(scoresLR, scoresMVG,  np.hstack(folds_labels), [0.5, 0.5])
        print(scores.shape)
        minDCF = dcf.compute_min_DCF(scores.ravel(), labels, 0.5, 1, 1)

import numpy
import matplotlib.pyplot as plt
import  pylab
import dcf 
import utility as util
import logistic_regression as lr
import svm
from tqdm import tqdm
from copy import deepcopy
from preprocessing import preprocess_Z_score
import matplotlib

# ======================================== FEATURES plots ==========================================
def plot_features_distr(D, labels, features, gau=False):
    n_features = len(features)
    _gau = "gau-" if gau else ""
    males = D[:, labels == 0]
    females = D[:, labels == 1]
    bins = 30

    for feature in range(n_features):
        plt.Figure()
        plt.xlabel(features[feature])
        
        dataset_m = males[feature, :]
        dataset_f = females[feature, :]

        plt.hist(dataset_m, bins=bins, density=True, label='male', alpha=0.4)
        plt.hist(dataset_f, bins=bins, density=True, label='female', alpha=0.4)

        plt.legend()
        plt.savefig(f"./plots/features/{_gau}/{features[feature]}.png", format="png")
        plt.show()
        
        

def plot_relation_beetween_feautures(D, labels, features):
    n_features = len(features)

    males = D[:, labels == 0]
    females = D[:, labels == 1]

    for featureA in range(n_features):
        for featureB in range(featureA, n_features):
            if featureA == featureB:
                continue
            plt.figure()
            plt.xlabel(labels[featureA])
            plt.ylabel(labels[featureB])

            plt.scatter(males[featureA, :], males[featureB, :], label='Male', alpha=0.4)
            plt.scatter(females[featureA, :], males[featureB, :], label='Female', alpha=0.4)

            plt.legend()
            plt.show()

# ============================================ CORRELATION between features plots ======================================================
def pearson_coeff(x, y):
    """
    Given two arrays evaluate the Pearson coefficient
    Parameters
    ---------
    x: numpy.array
        first array
    y: numpy.array
        second array
    """
    cov = numpy.cov(x, y)[0][1]
   
    x_var = numpy.var(x)
    y_var = numpy.var(y)

    return numpy.abs(cov / (numpy.sqrt(x_var) * numpy.sqrt(y_var)))

def plot_heatmap(D, features, color):
    """
    Plot the heatmap of a given dataset. This heat map will show the pearson coefficient between all the feauters.
    Parameters
    ---------
    D: dataset
    color: an optional value with the color of the heatmap
    """ 
    n_features = len(features)
    coeffs = numpy.zeros((n_features, n_features))


    # evaluate the person coefficient for each feature
    for i in range(n_features):
        for j in range(n_features):
            coeffs[i][j] = pearson_coeff(D[i, :], D[j, :])
    
    # plot the heat map 
    fig, ax = plt.subplots()
    im = ax.imshow(coeffs, interpolation='nearest', cmap=color)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # Loop over data dimensions and create text annotations.
    for i in range(len(coeffs)):
        for j in range(len(coeffs)):
            text = ax.text(j, i, numpy.around(coeffs[i, j],2),
                        ha="center", va="center", color="w")
    ax.set_title("Heat map")
    fig.tight_layout()
    plt.show()

# ================================================= MIN DCFs Plots ============================================================================
def compare_min_DCF_logreg(DTR, DTE, LTR, LTE, applications, quadratic=False, preprocessing=False, weighted=False):
    lambdas = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.3, 0.5, 1, 5, 10, 50, 100]
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    quadratic_ = 'quadratic' if quadratic else 'linear'
    colors = ['b', 'r', 'g']
    params = {
        'weighted' : weighted
    }
    
    max_y = 0
    DCFs_dict = dict()
    
    file_prefix = lr.compute_filename_prefix(quadratic, preprocessing, weighted)
    train_minDCFs, train_lambdas = lr.load_results(file_prefix)
    PATH = f"./plots/LogReg/experimental/{file_prefix}-minDCF.png"
    
    for i, application in enumerate(applications):
        pi, Cfn, Cfp = application
        params['priors'] = [pi, 1-pi]
        
        DCFs = lr.compute_minDCF_for_lambda(DTR, DTE, LTR, LTE, application, lambdas, quadratic, params)
        DCFs_dict[application] = DCFs
        max_y = max(max_y, numpy.amax(numpy.hstack((train_minDCFs[application], DCFs))))
        
        plt.plot(train_lambdas, train_minDCFs[application], color=colors[i], label=f"{app_labels[i]} [Val]", linestyle='dashed')
        plt.plot(lambdas, DCFs, color=colors[i], label=f"{app_labels[i]} [Eval]")
                                      
    plt.ylim(0, max_y + 0.05)
    plt.xscale('log')
    plt.title(f"DCF {quadratic_} logistic regression")
    plt.xlabel('lambda')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig(PATH, format='png')
    plt.show()
    return lambdas, DCFs_dict
    
def plot_min_DCF_logreg(folds, folds_labels, k, applications, quadratic=False, preprocessing=False, weighted=False):
    lambdas = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.3, 0.5, 1, 5, 10, 50, 100]
    #lambdas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 0.3, 0.5, 1, 5, 10]
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)']
    colors = ['b', 'r', 'g']
    max_y = 0
    
    quadratic_ = "quadratic" if quadratic else "linear"
    
    file_prefix = lr.compute_filename_prefix(quadratic, preprocessing, weighted)
    PATH = f"./plots/LogReg/{file_prefix}-minDCF.png"
     
    DCFs_dict = {}
    
    max_y = 0    
    for i, application in enumerate(applications):
        DCFs = []
        pi, Cfn, Cfp = application
        classPriors = [pi, 1-pi]
        for l in tqdm(lambdas):       
            if not quadratic:
                STE = util.k_folds(folds, folds_labels, k, lr.logreg, priors=classPriors, lambda_=l, preprocessing=preprocessing, weighted=weighted)
            else: 
                STE = util.k_folds(folds, folds_labels, k, lr.quadratic_logreg, priors=classPriors, lambda_=l, preprocessing=preprocessing, weighted=weighted)
            scores = numpy.hstack(STE)
            DCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            max_y = max(max_y, DCF)
            DCFs.append(DCF)
            DCFs_dict[application] = DCFs
        plt.plot(lambdas, DCFs, color=colors[i], label=app_labels[i])
            
    plt.ylim(0, max_y+0.1)
    plt.xscale('log')
    plt.title(f"DCF {quadratic_} logistic regression")
    plt.xlabel('lambda')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig(PATH, format='png')
    plt.show()
    return lambdas, DCFs_dict
# ================================================= MIN DCFs SVM Plots ============================================================================
def compare_min_DCF_svm(DTR, DTE, LTR, LTE, kernel:str, evaluation_points: tuple, balanced: bool, preprocessing: bool):
    #plot features
    #Cs = [0.005, 0.02,0.05, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 5, 10, 20, 50]
    Cs = [0.005, 0.05, 0.1, 0.5, 1, 5]
    
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] if kernel != 'rbf' else ['log(\u03BB)=-1', 'log(\u03BB)=-2', 'log(\u03BB)=-3'] 
    balanced_ = "balanced" if balanced else "not balanced"
    
    file_prefix = svm.compute_filename_prefix(balanced, preprocessing)
    train_minDCFs, train_Cs = svm.load_results(file_prefix, kernel)
    PATH = f"./plots/SVM/experimental/{kernel}-{file_prefix}-minDCF.png"
    
    max_y = 0
    minDCFs_dict = dict()
    
   
    for i, ep in enumerate(evaluation_points):
        DCFs = []
        if kernel == 'linear':
            pi, Cfn, Cfp = ep
            params = util.build_params(priors=[pi, 1-pi], balanced=balanced, kernel=kernel)
        elif kernel == 'poly':
            pi, Cfn, Cfp = ep
            params = util.build_params(priors=[pi, 1-pi], balanced=balanced, kernel=kernel, d=2, c=1,)
        elif kernel == 'rbf':
            params = util.build_params(priors=[0.5, 0.5], balanced=balanced, kernel=kernel, gamma=ep)
        
        minDCFs = svm.compute_minDCF_for_parameter(DTR, DTE, LTR, LTE, ep, Cs, params)
        minDCFs_dict[ep] = minDCFs
        max_y = max(max_y, numpy.amax(numpy.hstack((train_minDCFs[ep], minDCFs))))
        
        minDCFs = numpy.array(minDCFs).ravel()
        plt.plot(Cs, minDCFs, color=colors[i], label=f"{app_labels[i]} [Eval]")
        train_minDCF = numpy.array(train_minDCFs[ep]).ravel()
        plt.plot(train_Cs, train_minDCF, color=colors[i], label=f"{app_labels[i]} [Val]", linestyle='dashed' )
    
    plt.ylim(0, max_y+0.05)
    plt.title(f"minDCF for {kernel} SVM ({balanced_})")
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig(PATH, format="png")
    plt.show()
    return Cs, minDCFs_dict
        
        
        
def plot_min_DCF_svm(folds, folds_labels, k, applications, balanced=False, preprocessing=None):
    
    balanced_ = "balanced" if balanced else "not balanced" 
    preprocessing_ = preprocessing if preprocessing else "raw"
    
    PATH = f"./plots/SVM/{preprocessing_}-linear-{balanced_}-minDCF.png"
    
    Cs = [0.005, 0.02,0.05, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 5, 10, 20, 50]
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    minDCFs_dict = {}
    
    max_y = 0
    
    for i, application in enumerate(applications):
        DCFs = []
        pi, Cfn, Cfp = application
        classPriors = [pi, 1-pi]
        for C in tqdm(Cs):
            scores = util.k_folds(folds, folds_labels, k, svm.train_SVM_linear,SVM=True, C = C, balanced=balanced, preprocessing=preprocessing)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        minDCFs_dict[application] = DCFs.ravel()
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        
    plt.ylim(0, 1)
    plt.title(f"minDCF for linear SVM ({balanced_})")
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('DCF')
    plt.legend()
    plt.savefig(PATH, format="png")
    plt.show()
    return Cs, minDCFs_dict
    
def plot_min_DCF_poly_svm(folds, folds_labels, k, applications, degree=2.0, balanced=False, preprocessing=None):
    balanced_ = "balanced" if balanced else "not balanced" 
    preprocessing_ = "z-norm" if preprocessing else "raw"
    PATH = f"./plots/SVM/{preprocessing_}-poly{int(degree)}-{balanced_}-minDCF.png"
    Cs = [0.005, 0.05, 0.1, 0.5, 1, 5]
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    minDCFs_dict = {}
    for i, application in enumerate(applications):
        DCFs = []
        pi, Cfn, Cfp = application
        classPriors = [pi, 1-pi]
        for C in tqdm(Cs):
            scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM,  SVM=True, kernel='poly', C=C, d=degree, c=1, balanced=balanced, preprocessing=preprocessing)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        minDCFs_dict[application] = DCFs.ravel()
        plt.ylim(0, 1)
        plt.title(f"DCF for Poly(d={int(degree)}) SVM ({balanced_})")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        plt.legend()
    plt.savefig(PATH, format="png")
    plt.show()
    return Cs, minDCFs_dict

def plot_min_DCF_RBFsvm(folds, folds_labels, k, gammas, balanced=False, preprocessing=False):
    balanced_ = "balanced" if balanced else "not-balanced"
    preprocessing_ = "z-norm" if preprocessing else "raw"
    PATH = f"./plots/SVM/{preprocessing_}-RBF-{balanced_}-minDCF.png"
    Cs = [0.005, 0.01,0.02,0.05, 0.08, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 3, 5, 10, 20, 50]
    colors = ['b', 'r', 'g']
    app_labels = ['log(\u03B3)=-1', 'log(\u03B3)=-2', 'log(\u03B3)=-3'] 
    
    minDCFs_dict = {}
    
    for i,gamma in enumerate(gammas):
        DCFs = []
        pi, Cfn, Cfp = (0.5, 1, 1)
        classPriors = [pi, 1-pi]
        for C in tqdm(Cs):
            scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM, SVM=True, kernel='rbf', gamma=gamma, C=C, balanced=balanced, preprocessing=preprocessing)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        minDCFs_dict[gamma] = DCFs.ravel()
        plt.ylim(0, 1)
        plt.title("DCF for RBF kernel SVM")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        plt.legend()
    plt.savefig(PATH, format="png")
    plt.show()
    
    return Cs, minDCFs_dict
# ================================================= MIN DCFs GMM Plots ============================================================================

def plot_minDCF_GMM_hist(DCFs_list: list, G: int, labels: list, filename='plot', experimental= False, title="", colors=['lightsalmon', 'orangered', 'gold', 'orange']):
    x_labels = list(map(lambda val:2**val, range(G)))
    x = numpy.arange(len(x_labels))
    width = 0.18
    _experimental = "experimental/" if experimental else ""
    path = f"./plots/GMM/{_experimental}{filename}.png"
    
    n_hists = len(DCFs_list)
    offsets = list( range(-int(n_hists/2) - 1, int(n_hists/2) + 2, 2))
    print("n_hist:", n_hists, "offsets", offsets)
    
    fig, ax = plt.subplots()
    for DCFs, offset, label, color in zip(DCFs_list, offsets, labels, colors):
        ax.bar(x + offset*width/2, DCFs, width, label=label, color=color)
    
    ax.set_ylabel('DCF')
    ax.set_xticks(x, x_labels)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(path, format='png')

    plt.show()




# ================================================================ DET Plot ===================================================================
def plot_DET(llrs:list, L: numpy.array, plot_labels:list, colors: list =['r', 'b', 'm', 'g', 'y'], save_figure:bool = True, training:bool = True, multiple_labels: bool = False):
    training_ = "training" if training else "experimental"
    models = "-".join(plot_labels)
    PATH = f"./plots/evaluation/{training_}/DET_{models}.png"
    
    fig,ax = plt.subplots()
    
    if not multiple_labels:
        for llr, plot_label, color in zip(llrs, plot_labels, colors):     
            print(plot_label)
            DET_points_FNR, DET_points_FPR = compute_DET_points(llr, L)
            ax.plot(DET_points_FNR, DET_points_FPR, color=color, label=plot_label)
    else:
        for llr, lbl, plot_label, color in zip(llrs, L, plot_labels, colors):     
            DET_points_FNR, DET_points_FPR = compute_DET_points(llr, lbl)
            ax.plot(DET_points_FNR, DET_points_FPR, color=color, label=plot_label)
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR")
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.legend()
    if save_figure:
        plt.savefig(PATH, format='png')
    plt.show()

def compute_DET_points(llr, L):
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    DET_points_FNR = numpy.zeros(L.shape[0] +2 )
    DET_points_FPR = numpy.zeros(L.shape[0] +2 )
    for (idx,t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        FNR = 1 - (numpy.bitwise_and(pred == 1, L == 1 ).sum() / N_label1)
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / N_label0
        DET_points_FNR[idx] = FNR
        DET_points_FPR[idx] = FPR
    return DET_points_FNR, DET_points_FPR
# =============================================== ROC Plots ==================================================
def plot_ROC(llrs: list, labels: list, plot_labels: list, save_figure:bool = True, training:bool = True):
    training_ = "training" if training else "experimental"
    models = "-".join(plot_labels)
    PATH = f"./plots/evaluation/{training_}/ROC_{models}.png"
    
    for llr, plot_label in zip(llrs, plot_labels):
        ROC_points_TPR, ROC_points_FPR = compute_ROC_points(llr, labels)
        plt.plot(ROC_points_FPR, ROC_points_TPR, label=plot_label)
        
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    
    if save_figure:
        plt.savefig(PATH, format='png')
    
    plt.show()

def compute_ROC_points(llr, L):
    tresholds = numpy.concatenate([numpy.array([-numpy.inf]),numpy.sort(llr),numpy.array([numpy.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    ROC_points_TPR = numpy.zeros(L.shape[0] +2 )
    ROC_points_FPR = numpy.zeros(L.shape[0] +2 )
    for (idx,t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        TPR = numpy.bitwise_and(pred == 1, L == 1 ).sum() / N_label1
        FPR = numpy.bitwise_and(pred == 1, L == 0).sum() / N_label0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR

# =========================================================== Bayes Error Plot =============================================================
def bayes_error_plot(llrs: list, labels: list, plot_labels: list, log_regs: list, n_points:int = 100, colors: list = ['r', 'b', 'g', 'm', 'y'], save_figure: bool = True, training:bool = True, calibrated: bool = False, multiple_labels:bool = False):
    training_ = "training" if training else "experimental"
    models = "-".join(plot_labels)
    calibrated_ = "-calibrated" if calibrated else ""
    PATH = f"./plots/evaluation/{training_}/BEP_{models}{calibrated_}.png"
    
    max_y = 0
    if not multiple_labels:
        for llr, plot_label, log_reg, color in zip(llrs, plot_labels, log_regs, colors):
            p_array = numpy.linspace(-3, 3, n_points)
            minDCFs = dcf.bayes_error_points(p_array, llr, labels, True, log_reg)
            max_y = max(max_y, numpy.max(minDCFs))
            actDCFs = dcf.bayes_error_points(p_array, llr, labels, False, log_reg)
            max_y = max(max_y, numpy.max(actDCFs))
            plt.plot(p_array, minDCFs, label=f"{plot_label} minDCF", color=color, linestyle='dashed')
            plt.plot(p_array, actDCFs, label=f"{plot_label} actDCF", color=color)
    else:
        for llr, lbl, plot_label, log_reg, color in zip(llrs, labels, plot_labels, log_regs, colors):
            p_array = numpy.linspace(-3, 3, n_points)
            minDCFs = dcf.bayes_error_points(p_array, llr, lbl, True, log_reg)
            max_y = max(max_y, numpy.max(minDCFs))
            actDCFs = dcf.bayes_error_points(p_array, llr, lbl, False, log_reg)
            max_y = max(max_y, numpy.max(actDCFs))
            plt.plot(p_array, minDCFs, label=f"{plot_label} minDCF", color=color, linestyle='dashed')
            plt.plot(p_array, actDCFs, label=f"{plot_label} actDCF", color=color)
        
    title = "Bayes Error Plot"
    
    plt.yticks(numpy.arange(0, min(max_y+0.1, 1), 0.05))
    plt.title(title)
    plt.legend()
    
    if save_figure:
        plt.savefig(PATH, format='png')
    
    plt.show()
    
        
        
        
import numpy
import matplotlib.pyplot as plt
import matplotlib
import  pylab
import dcf 
import utility as util
import logistic_regression as lr
import svm
def plot_features_distr(D, labels, features):
    n_features = len(features)

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


def plot_min_DCF_logreg(folds, folds_labels, k, applications, quadratic=False):
    lambdas = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 8e-3, 1e-2, 2e-2, 5e-2, 1e-1, 0.3, 0.5, 1, 5, 10, 50, 100]
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    #print("lambdas:", lambdas)
    for i, application in enumerate(applications):
        DCFs = []
        pi = application[0]
        Cfn = application[1]
        Cfp = application[2]
        classPriors = [pi, 1-pi]
        for l in lambdas:
            print("for lambda:", l)
            if quadratic:
                STE = util.k_folds(folds, folds_labels, k, lr.logreg, priors=classPriors, lambda_=l)
            else:
                STE = util.k_folds(folds, folds_labels, k, lr.quadratic_logreg, priors=classPriors, lambda_=l)
            scores = numpy.hstack(STE)
            DCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(DCF)
        plt.ylim(0, 1)
        plt.xscale('log')
        plt.xlabel('lambda')
        plt.ylabel('DCF')
        plt.plot(lambdas, DCFs, color=colors[i], label=app_labels[i])
        plt.legend()
    plt.show()

def plot_min_DCF_svm(folds, folds_labels, k, applications, balanced=False):
    balanced_ = "balanced" if balanced else "not balanced" 
    PATH = f"./plots/SVM/linear-{balanced_}-minDCF.png"
    Cs = [0.005, 0.02,0.05, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 5, 10, 20, 50]
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    for i, application in enumerate(applications):
        DCFs = []
        pi, Cfn, Cfp = application
        classPriors = [pi, 1-pi]
        for C in Cs:
            scores = util.k_folds(folds, folds_labels, k, svm.train_SVM_linear, C = C, balanced=balanced)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        plt.ylim(0, 1)
        
        plt.title(f"minDCF for linear SVM ({balanced_})")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        plt.legend()
    plt.save_fig(PATH, format="png")
    plt.show()
    
def plot_min_DCF_poly_svm(folds, folds_labels, k, applications, degree=2.0, balanced=False):
    balanced_ = "balanced" if balanced else "not balanced" 
    PATH = f"./plots/SVM/poly{int(degree)}-{balanced_}-minDCF.png"
    Cs = [0.005, 0.02,0.05, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 5, 10, 20, 50]
    colors = ['b', 'r', 'g']
    app_labels = ['minDCF(pi=0.5)', 'minDCF(pi=0.1)', 'minDCF(pi=0.9)'] 
    for i, application in enumerate(applications):
        DCFs = []
        pi, Cfn, Cfp = application
        classPriors = [pi, 1-pi]
        for C in Cs:
            scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM, kernel='poly', C=C, d=degree, c=1)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        plt.ylim(0, 1)
        plt.title(f"DCF for Poly(d={int(degree)}) SVM ({balanced_})")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        plt.legend()
    plt.save_fig(PATH, format="png")
    plt.show()

def plot_min_DCF_RBFsvm(folds, folds_labels, k, gammas):
    PATH = "./plots/SVM/RBF-minDCF.png"
    Cs = [0.005, 0.01,0.02,0.05, 0.08, 0.10, 0.20, 0.30, 0.5, 0.8, 1, 3, 5, 10, 20, 50]
    colors = ['b', 'r', 'g']
    app_labels = ['log(\u03BB)=-1', 'log(\u03BB)=-2', 'log(\u03BB)=-3'] 
    for i in range(3):
        DCFs = []
        pi, Cfn, Cfp = (0.5, 1, 1)
        classPriors = [pi, 1-pi]
        for C in Cs:
            scores = util.k_folds(folds, folds_labels, k, svm.train_non_linear_SVM, kernel='rbf', gamma=gammas[i], C=C)
            scores = numpy.hstack(scores)
            minDCF = dcf.compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
            DCFs.append(minDCF)
        DCFs = numpy.array(DCFs)
        plt.ylim(0, 1)
        plt.title("DCF for RBF kernel SVM")
        plt.xscale('log')
        plt.xlabel('C')
        plt.ylabel('DCF')
        plt.plot(Cs, DCFs.ravel(), color=colors[i], label=app_labels[i])
        plt.legend()
    plt.save_fig(PATH, format="png")
    plt.show()

def plot_minDCF_GMM_hist(DCFsA, DCFsB, G, filename='plot'):
    labels = list(map(lambda val:2**val, range(G)))
    x = numpy.arange(len(labels))
    width = 0.35
    path = f"./plots/GMM/{filename}.png"
    

    fig, ax = plt.subplots()
    ax.bar(x - width/2, DCFsA, width, label='GMM with Z-normalization')
    ax.bar(x + width/2, DCFsB, width, label='GMM with raw features')
    ax.set_ylabel('DCF')
    ax.set_xticks(x, labels)
    ax.legend()

    fig.tight_layout()
    plt.savefig(path, format='png')

    plt.show()
    
def DETCurve(fps,fns):
    """
    Given false positive and false negative rates, produce a DET Curve.
    The false positive rate is assumed to be increasing while the false
    negative rate is assumed to be decreasing.
    """
    axis_min = min(fps[0],fns[-1])
    fig,ax = plt.subplots()
    plt.plot(fps,fns)
    plt.yscale('log')
    plt.xscale('log')
    ticks_to_use = [0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50]
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks(ticks_to_use)
    ax.set_yticks(ticks_to_use)
    plt.axis([0.001,50,0.001,50])
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

def plot_ROC(llr, L):
    ROC_points_TPR, ROC_points_FPR = compute_ROC_points(llr, L)
    plt.plot(ROC_points_FPR, ROC_points_TPR, color='r')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.grid()
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

def plot_DET3(llr, L):
    DET_points_TPR1, DET_points_FPR1 = compute_DET_points(llr, L)
    
    plt.plot(DET_points_FPR1, DET_points_TPR1, color='r')

    plt.xlabel("FPR")
    plt.ylabel("FNR")
    
    plt.xscale('log')
   
    plt.legend()
    plt.grid()
    plt.show()
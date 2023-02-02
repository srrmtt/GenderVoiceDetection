import numpy
import scipy
import pylab
import utility as util


def compute_conf_matrix_binary(prediction:numpy.array, labels:numpy.array):
    """
    Compute confusion matrix from prediction array and ground truth labels for a binary problem.
    """
    C = numpy.zeros((2,2))

    C[0,0] = ((prediction == 0) * (labels == 0)).sum()
    C[0,1] = ((prediction == 0) * (labels == 1)).sum()
    C[1,0] = ((prediction == 1) * (labels == 0)).sum()
    C[1,1] = ((prediction == 1) * (labels == 1)).sum()
    return C 

def assign_labels(scores:numpy.array, pi:float, Cfn:float, Cfp:float, th:float=None):
    """
    Assign a labels given the scores and an optional treshold.
    """
    if th is None:
        th = -numpy.log(pi * Cfn) + numpy.log((1-pi) * Cfn)
    P = scores > th
    return numpy.int32(P)

def compute_emp_bayes_binary(conf_matrix:numpy.array, pi:float, Cfn:float, Cfp:float):
    fnr = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1,1])
    fpr = conf_matrix[1,0] / (conf_matrix[0,0] + conf_matrix[1,0])

    return pi * Cfn * fnr + (1-pi) * Cfp * fpr

def compute_normalized_emp_bayes(conf_matrix: numpy.array, pi:float, Cfn:float, Cfp:float):
    """
    compute the normalized empirical bayes error given the confusion matrix and the application.
    """
    empBayes = compute_emp_bayes_binary(conf_matrix, pi, Cfn, Cfp)
    return empBayes / min(pi*Cfn, (1-pi) * Cfp)

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    '''
    this function assumes that the scores are log likelihood ratios.
    '''
    pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    conf_matrix = compute_conf_matrix_binary(pred, labels)
    return compute_normalized_emp_bayes(conf_matrix, pi, Cfn, Cfp)

def compute_min_DCF(scores:numpy.array, labels:numpy.array, pi:float, Cfn:float, Cfp:float):
    '''
    Compute the min DCF metric given an array of scores and a parallel array of labels.

    '''
    t = numpy.array(scores)
    t.sort()
    t = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []

    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th= _th))
    minDCF = numpy.array(dcfList).min()
    return minDCF

def bayes_error_points(pArray, scores, labels, minCost=False, log_reg=False):
    """
    computes the points for the bayes error plot. The last parameters is optional and tell if the scores come
    from a logistic regression method. 
    """
    y = []
    _correction = 0
    if log_reg:
        pi = scores[labels == 0].shape[0] / scores.shape[0]
        _correction = numpy.log(pi / (1 - pi))
        print(f"correction enabled: {_correction}, pi: {pi},{scores.shape[0]}, {scores[labels == 0].shape[0]}")
    for p in pArray:
        pi = 1.0 / (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1) - _correction)
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1) - _correction)
    return numpy.array(y) 
  # for linear regression we have to subtract the term log(pi / (1 - pi)) where pi = #class1 / #training samples

def GMM_minDCF(folds_component_llrs, folds_labels, G, k, application):
    """
    Compute the min DCF for the GMM results, this function has been implemented due to the the specific llrs GMM 
    structure.
    """
    pi, Cfn, Cfp = application
    print("llrs:", folds_component_llrs.shape)
    minDCFs = []
    # iterate over components
    for i in range(G):
        scores = []
        # take the fold of the k-fold method correponding to the same component
        for j in range(k):
            scores.append(folds_component_llrs[j][i])
        if k > 0:
            labels = numpy.hstack(folds_labels)
            scores = numpy.hstack(scores)
        else:
            scores = folds_component_llrs[i]
            labels = folds_labels
        minDCF = compute_min_DCF(scores, labels, 0.5, 1, 1)
        minDCFs.append(minDCF)
    minDCFs = numpy.array(minDCFs)
    return minDCFs

def compute_act_min_DCFs_from_scores(models_scores: list, labels: numpy.array, applications: list, models_names: list):
    """
    Print the act DCF and the min DCF, given the models scores and a their names, it's just a fancy print.
    """
    for application in applications:
        pi, Cfn, Cfp = application
        print(f"application: {application}")
        for scores, model_name in zip(models_scores, models_names):
            act_dcf = compute_act_DCF(scores, labels, pi, Cfn, Cfp)
            min_dcf = compute_min_DCF(scores, labels, pi, Cfn, Cfp)
            print(f"\tActual DCF for {model_name}: {act_dcf}")
            print(f"\tMin DCF for {model_name}: {min_dcf}")
            print("\t------------------------------------")
        
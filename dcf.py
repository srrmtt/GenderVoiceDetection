import numpy
import scipy
import pylab
import utility as util

def confusion_matrix(prediction, labels):
    conf_matrix = numpy.zeros((2,2))
    for i in [0, 1]:
        for j in [0, 1]:
            conf_matrix[i,j] = ((prediction == i) * (labels == j)).sum()
    return conf_matrix
def compute_conf_matrix_binary(prediction, labels):
    C = numpy.zeros((2,2))

    C[0,0] = ((prediction == 0) * (labels == 0)).sum()
    C[0,1] = ((prediction == 0) * (labels == 1)).sum()
    C[1,0] = ((prediction == 1) * (labels == 0)).sum()
    C[1,1] = ((prediction == 1) * (labels == 1)).sum()
    return C 

def assign_labels(scores, pi, Cfn, Cfp, th=None):
    if th is None:
        th = -numpy.log(pi * Cfn) + numpy((1-pi) * Cfn)
    P = scores > th
    return numpy.int32(P)

def compute_emp_bayes_binary(conf_matrix , pi, Cfn, Cfp):
    fnr = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1,1])
    fpr = conf_matrix[1,0] / (conf_matrix[0,0] + conf_matrix[1,0])

    return pi * Cfn * fnr + (1-pi) * Cfp * fpr

def compute_normalized_emp_bayes(conf_matrix, pi, Cfn, Cfp):
    empBayes = compute_emp_bayes_binary(conf_matrix, pi, Cfn, Cfp)
    return empBayes / min(pi*Cfn, (1-pi) * Cfp)

def compute_act_DCF(scores, labels, pi, Cfn, Cfp, th=None):
    '''
    this function assumes that the scores are log likelihood ratios.
    '''
    pred = assign_labels(scores, pi, Cfn, Cfp, th=th)
    conf_matrix = compute_conf_matrix_binary(pred, labels)
    return compute_normalized_emp_bayes(conf_matrix, pi, Cfn, Cfp)

def compute_min_DCF(scores, labels, pi, Cfn, Cfp):
    t = numpy.array(scores)
    t.sort()
    t = numpy.concatenate([numpy.array([-numpy.inf]), t, numpy.array([numpy.inf])])
    dcfList = []

    for _th in t:
        dcfList.append(compute_act_DCF(scores, labels, pi, Cfn, Cfp, th= _th))
    minDCF = numpy.array(dcfList).min()
    #print("minDCF:", minDCF)
    return minDCF

def bayes_error_plot(pArray, scores, labels, minCost=False):
    y = []
    for i in pArray:
        pi = 1.0 / (1.0 + numpy.exp(-p))
        if minCost:
            y.append(compute_min_DCF(scores, labels, pi, 1, 1))
        else:
            y.append(compute_act_DCF(scores, labels, pi, 1, 1))
    return numpy.array(y) 
  # for linear regression we have to subtract the term log(pi / (1 - pi)) where pi = #class1 / #training samples

def GMM_minDCF(folds_component_llrs, folds_labels, G, k, application):
    pi, Cfn, Cfp = application
    minDCFs = []
    for i in range(G):
        scores = []
        for j in range(k):
            scores.append(folds_component_llrs[j][i])
        scores = numpy.hstack(scores)
        print("scores", scores.shape, "labels",numpy.hstack(folds_labels).shape)
        minDCF = compute_min_DCF(scores, numpy.hstack(folds_labels), pi, Cfn, Cfp)
        print("\t\t[", 2**i,"] components")
        print("\t\t\tminDCF:", minDCF)
        minDCFs.append(minDCF)
    minDCFs = numpy.array(minDCFs)
    return minDCFs
  
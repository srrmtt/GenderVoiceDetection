import numpy
import matplotlib.pyplot as plt

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
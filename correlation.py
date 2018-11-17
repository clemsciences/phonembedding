import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cossim
from scipy.stats import pearsonr


def get_similarity_matrix(embedded_vectors, embedded_chars):
    """

    :param embedded_vectors:
    :param embedded_chars:
    :return:
    """
    correlation_matrix = np.zeros((len(embedded_chars), len(embedded_chars)))

    for i, embedded_vector1 in enumerate(embedded_vectors):
        if i not in embedded_chars or embedded_vector1 is None:
            continue
        for j, embedded_vector2 in enumerate(embedded_vectors):
            if j not in embedded_chars or embedded_vector2 is None:
                continue
            correlation_matrix[i][j] = cossim(embedded_vector1.reshape(1, -1), embedded_vector2.reshape(1, -1))
    return correlation_matrix


def correlation(x, y):
    """

    :param x:
    :param y:
    :return: Pearson's R correlation of flattened values of x and y
    """
    assert (x.shape == y.shape)

    xl = []
    yl = []

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            xl.append(x[i][j])
            yl.append(y[i][j])
    return pearsonr(xl, yl)

# from sklearn.metrics.pairwise import cosine_similarity as cossim
import numpy as np

WINDOW = 2


def build_matrix(data, charencoder):
    """

    :param data:
    :param charencoder:
    :return:
    """
    # Initializes to non-zero to avoid NAN during np.log.
    joint_counts = np.ones((len(charencoder), len(charencoder))) * 0.0001
    single_counts = np.ones((len(charencoder), 1)) * 0.0001

    joint_total = 0
    single_total = 0

    for wf, _, _ in data:
        for i, c in enumerate(wf):
            for j in range(i - WINDOW, i + WINDOW):
                if j == i or j < 0 or j >= len(wf):
                    continue
                joint_counts[c][wf[j]] += 1
                joint_total += 1
            single_counts[c][0] += 1
            single_total += 1

    joint_distribution = joint_counts * (1.0 / joint_total)
    single_distribution = single_counts * (1.0 / single_total)
    pmi = np.log(np.divide(joint_distribution, np.dot(single_distribution, np.transpose(single_distribution))))
    return np.multiply(pmi, pmi > 0)


def get_svd(data, charencoder):
    """

    :param data: list of list of
    :param charencoder:
    :return:
    """
    ppmi_matrix = build_matrix(data, charencoder)
    u, s, vt = np.linalg.svd(ppmi_matrix)
    return np.dot(u, np.diag(s))


def truncate(m, d):
    return m[0:m.shape[0], 0:d]

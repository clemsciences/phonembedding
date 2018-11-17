# from sklearn.metrics.pairwise import cosine_similarity as cossim
import numpy as np

WINDOW = 2


def build_matrix(data, character_encoder):
    """

    :param data:
    :param character_encoder:
    :return:
    """
    # Initializes to non-zero to avoid NAN during np.log.
    joint_counts = np.ones((len(character_encoder), len(character_encoder))) * 0.0001
    single_counts = np.ones((len(character_encoder), 1)) * 0.0001

    joint_total = 0
    single_total = 0

    for word_form, _, _ in data:
        for i, character in enumerate(word_form):
            for j in range(i - WINDOW, i + WINDOW):
                if j == i or j < 0 or j >= len(word_form):
                    continue
                joint_counts[character][word_form[j]] += 1
                joint_total += 1
            single_counts[character][0] += 1
            single_total += 1

    joint_distribution = joint_counts * (1.0 / joint_total)
    single_distribution = single_counts * (1.0 / single_total)
    pmi = np.log(np.divide(joint_distribution, np.dot(single_distribution, np.transpose(single_distribution))))
    return np.multiply(pmi, pmi > 0)


def get_svd(data, character_encoder):
    """

    :param data: list of list of
    :param character_encoder:
    :return:
    """
    ppmi_matrix = build_matrix(data, character_encoder)
    u, s, vt = np.linalg.svd(ppmi_matrix)
    return np.dot(u, np.diag(s))


def truncate(m, d):
    return m[0:m.shape[0], 0:d]

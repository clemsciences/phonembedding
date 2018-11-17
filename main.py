import numpy as np
import scipy.stats as st
from random import seed

from data_manager import read_data
from svd import get_svd
from w2v import get_w2v
from correlation import get_similarity_matrix, correlation
from features import getphonfeatures
from rnn import initmodel, encode, decode, update, train

seed(42)
np.random.seed(43)

# To test 
N = 20
P = 0.99


def confidence_interval_value(a):
    return st.t.interval(P, len(a) - 1, loc=np.mean(a), scale=st.sem(a))


def truncate(m, d):
    return m[0:m.shape[0], 0:d]


def matshuf(m):
    res = np.array(m)
    np.random.shuffle(res)
    return res


def get_svd_embeddings(filename, character_encoder, embedded_chars, character_decoder):
    """

    :param filename: filename of data
    :param character_encoder:
    :param embedded_chars: useless, but useful for function compatibility
    :param character_decoder: useless, but useful for function compatibility
    :return:
    """
    svd_embedding = get_svd(filename, character_encoder)
    svd_embeddings = [truncate(svd_embedding, n) for n in [5, 15, 30]]
    return svd_embeddings


# def getw2vembs(data, character_encoder, embedded_chars, character_decoder):
#     """
#
#     :param data:
#     :param character_encoder: useless, but useful for function compatibility
#     :param embedded_chars:
#     :param character_decoder:
#     :return:
#     """
#     return [get_w2v(data, embedded_chars, 5, character_decoder),
#             get_w2v(data, embedded_chars, 15, character_decoder),
#             get_w2v(data, embedded_chars, 30, character_decoder)]


def check_r(interval_value, r):
    if interval_value[0] < r and interval_value[1] < r:
        return "<"
    else:
        return ">"


def correlation_experiment(filename, language, embedding_function, name):
    data, character_encoder, tag_encoder, embedded_chars = read_data(filename, language)
    character_decoder = {v: k for k, v in character_encoder.items()}
    features = getphonfeatures()
    language_features = [np.array(features[character_decoder[f]])
                         if character_decoder[f] in features
                         else None for f in range(len(character_encoder))]

    feature_similarity = get_similarity_matrix(language_features, embedded_chars)

    embeddings = embedding_function(data, character_encoder, embedded_chars, character_decoder)

    similarities = [get_similarity_matrix(m, embedded_chars) for m in embeddings]
    rs = [correlation(feature_similarity, similarities[i])[0] for i in [0, 1, 2]]
    print("%s %s:" % (language, name))
    print(" PEARSON R FOR EMBEDDING AND FEATURE REPR. SIMILARITIES:")
    print("  %s,DIM=5" % language, rs[0])
    print("  %s,DIM=15" % language, rs[1])
    print("  %s,DIM=30" % language, rs[2])

    random_rs = [[], [], []]
    for i in range(N):
        random_embeddings = [matshuf(m) for m in embeddings]
        random_similarities = [get_similarity_matrix(m, embedded_chars) for m in random_embeddings]
        random_rs[0].append(correlation(feature_similarity, random_similarities[0])[0])
        random_rs[1].append(correlation(feature_similarity, random_similarities[1])[0])
        random_rs[2].append(correlation(feature_similarity, random_similarities[2])[0])

    print((" P=%.2f CONF. INTERVALS FOR PEARSON R OF RANDOM ASSIGNMENT OF\n" % P) +
          " EMBEDDINGS TO PHONEMES AND PHONETIC FEATURE DESCRIPTIONS:")
    civals = [confidence_interval_value(random_rs[i]) for i in [0, 1, 2]]
    print("  %s,DIM=5" % language, confidence_interval_value(random_rs[0]), check_r(civals[0], rs[0]), rs[0])
    print("  %s,DIM=15" % language, confidence_interval_value(random_rs[1]), check_r(civals[1], rs[1]), rs[1])
    print("  %s,DIM=30" % language, confidence_interval_value(random_rs[2]), check_r(civals[2], rs[2]), rs[2])
    print()


if __name__ == "__main__":
    print("1. CORRELATION EXPERIMENTS")
    print("--------------------------")
    print()
    correlation_experiment("data/finnish", "FI", get_svd_embeddings, "SVD")
    # correlation_experiment("data/finnish", "FI", getw2vembs, "W2V")
    #
    # correlation_experiment("data/spanish", "ES", getsvdembs, "SVD")
    # correlation_experiment("data/spanish", "ES", getw2vembs, "W2V")
    #
    # correlation_experiment("data/turkish", "TUR", getsvdembs, "SVD")
    # correlation_experiment("data/turkish", "TUR", getw2vembs, "W2V")

    # TODO getrnnembs is missing
    # correlation_experiment("data/finnish", "FI", getrnnembs, "RNN")
    # correlation_experiment("data/turkish", "TUR", getrnnembs, "RNN")
    # correlation_experiment("data/spanish", "ES", getrnnembs, "RNN")

    training_data, training_character_encoder, training_tag_encoder, training_embedded_characters = \
        read_data('data/finnish', "FI")
    training_modeld = initmodel(training_character_encoder, training_tag_encoder, 15)
    training_encoded = encode(training_data[0][1], training_data[0][2], training_modeld)
    train(training_data, training_modeld)
    for _ in range(100):
        print(update(training_data[0][1], training_data[0][2], training_data[0][0], training_modeld))

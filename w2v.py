import numpy as np

from gensim.models import Word2Vec

WINDOW = 1
SG = 1
HS = 0
MINCOUNT = 100
NS = 3


def get_w2v(data, embedded_characters, dim, character_decoder):
    """

    :param data:
    :param embedded_characters:
    :param dim:
    :param character_decoder: useless, but useful for compatibility reason
    :return:
    """
    # Encode ids to strings for gensim implementation of w2v.
    words = [[str(c) for c in wf] for wf, lemma, tags in data]
    lemmas = [[str(c) for c in lemma] for wf, lemma, tags in data]
    model = Word2Vec(sentences=words + lemmas,
                     size=dim,
                     window=WINDOW,
                     sg=SG,
                     hs=HS,
                     min_count=MINCOUNT,
                     negative=NS)
    m = np.zeros((len(embedded_characters), dim))
    for i in range(len(embedded_characters)):
        m[i] = model[str(i)]
    return m

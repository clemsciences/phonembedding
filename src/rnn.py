from copy import copy

from torch import FloatTensor, LongTensor, randn, stack, cat
from torch import max as tmax
from torch.autograd import Variable
from torch.nn import Embedding, Sequential, Linear, LogSoftmax, NLLLoss, LSTM, ModuleList
from torch.optim import Adam

import numpy as np

DTYPE = FloatTensor
LTYPE = LongTensor
LSTMDIM = 32
MAXWFLEN = 40
LOSS = NLLLoss()
LEARNINGRATE = 0.001
BETAS = (0.9, 0.9)
EPOCHS = 20

BD = "<#>"


def initmodel(character_encoder, tag_encoder, embedded_dimension):
    """

    :param character_encoder:
    :param tag_encoder:
    :param embedded_dimension:
    :return:
    """
    character_encoder = copy(character_encoder)
    tag_encoder = copy(tag_encoder)
    tag_encoder[BD] = len(tag_encoder)
    character_encoder[BD] = len(character_encoder)
    character_embedding = Embedding(len(character_encoder), embedded_dimension)
    tag_embedding = Embedding(len(tag_encoder), embedded_dimension)
    encoder_part = LSTM(input_size=embedded_dimension,
                        hidden_size=LSTMDIM,
                        num_layers=1,
                        bidirectional=1).type(DTYPE)
    ench0 = randn(2, 1, LSTMDIM).type(DTYPE)
    encc0 = randn(2, 1, LSTMDIM).type(DTYPE)

    decoder_part = LSTM(input_size=2 * LSTMDIM + embedded_dimension,
                        hidden_size=LSTMDIM,
                        num_layers=1).type(DTYPE)
    dech0 = randn(2, 1, 2 * LSTMDIM + embedded_dimension).type(DTYPE)
    decc0 = randn(2, 1, 2 * LSTMDIM + embedded_dimension).type(DTYPE)

    pred = Linear(LSTMDIM, len(character_encoder)).type(DTYPE)
    softmax = LogSoftmax().type(DTYPE)

    model = ModuleList([character_embedding,
                        tag_embedding,
                        encoder_part,
                        decoder_part,
                        pred,
                        softmax])
    optimizer = Adam(model.parameters(),
                     lr=LEARNINGRATE,
                     betas=BETAS)

    return {'model': model,
            'optimizer': optimizer,
            'cencoder': character_encoder,
            'tencoder': tag_encoder,
            'cembedding': character_embedding,
            'tembedding': tag_embedding,
            'enc': encoder_part,
            'ench0': ench0,
            'encc0': encc0,
            'dec': decoder_part,
            'dech0': dech0,
            'decc0': decc0,
            'pred': pred,
            'sm': softmax,
            'embdim': embedded_dimension}


def encode(lemma, tags, modeldict):
    """

    :param lemma:
    :param tags:
    :param modeldict:
    :return:
    """
    lemma = [modeldict['cencoder'][BD]] + lemma
    tags = tags + [modeldict['cencoder'][BD]]
    lemma_length = len(lemma)
    tags_length = len(tags)
    lemma = Variable(LTYPE(lemma), requires_grad=0)
    tags = Variable(LTYPE(tags), requires_grad=0)
    lemma = modeldict['cembedding'](lemma)
    tags = modeldict['tembedding'](tags)

    _all = cat([lemma, tags], dim=0)
    _, finals = modeldict['enc'](_all.view(lemma_length + tags_length, 1, modeldict['embdim']),
                                 (Variable(modeldict['ench0'], requires_grad=1),
                                  Variable(modeldict['encc0'], requires_grad=1)))
    finalh0, finalc0 = finals
    return finalc0


def decode(encoded, modeldict, wflen=MAXWFLEN):
    """

    :param encoded:
    :param modeldict:
    :param wflen:
    :return:
    """
    pchar = modeldict['cencoder'][BD]
    h = Variable(modeldict['ench0'], requires_grad=1)
    c = Variable(modeldict['encc0'], requires_grad=1)
    cdistrs = []
    chars = []
    for i in range(min(wflen + 1, MAXWFLEN)):
        pchar = Variable(LTYPE([pchar]), requires_grad=0)
        pemb = modeldict['cembedding'](pchar)
        _input = cat([pemb, encoded.view(1, 2 * LSTMDIM)], dim=1)
        _, states = modeldict['dec'](_input, (h, c))
        h, c = states
        c = c.view(1, LSTMDIM)
        cdistr = modeldict['sm'](modeldict['pred'](c))
        cdistrs.append(cdistr)
        _, pchar = tmax(cdistr, 1)
        pchar = int(pchar.data.numpy()[0])
        chars.append(pchar)
    return cat(cdistrs, dim=0), chars


def update(lemma, tags, word_form, modeldict):
    """

    :param lemma:
    :param tags:
    :param word_form:
    :param modeldict:
    :return:
    """
    encoded = encode(lemma, tags, modeldict)
    print(encoded.shape)
    cdistrs, chars = decode(encoded, modeldict, len(word_form))
    cdistrs = cdistrs[:len(chars)]
    chars = Variable(LTYPE(chars), requires_grad=0)
    loss = LOSS(cdistrs, chars)
    lossval = loss.data[0]
    loss.backward()
    modeldict['optimizer'].step()
    return lossval


def train(data, modeldict):
    """

    :param data:
    :param modeldict:
    :return:
    """
    for i in range(EPOCHS):
        totloss = 0
        for j, wlt in enumerate(data):
            wf, lemma, tags = wlt
            totloss += update(lemma, tags, wf, modeldict)
            print(i + 1, " ", len(data), " ", totloss / (i + 1))

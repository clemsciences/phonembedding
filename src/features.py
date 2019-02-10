"""

"""

VOWELS = {'a', 'o', 'u', 'i', 'ä', 'ö', 'Y', 'e', 'ı', 'é'}
SPREPLACE = {'b': 'B', 'd': 'D', 'g': 'G'}

features = ['cons', 'son', 'syll', 'voice', 'labial', 'coronal', 'dorsal',
            'pharyngeal', 'lateral', 'nasal', 'conti', 'delayrelease',
            'front', 'back', 'high', 'low', 'rounding', 'tense', 'distr',
            'tap', 'ant', 'strid']


def getphonfeatures():
    """

    :return: feature vectors of IPA characters
    """
    phoneme_to_feature = {}

    phoneme_to_feature['p'] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['b'] = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['t'] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    phoneme_to_feature['d'] = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    phoneme_to_feature['k'] = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['g'] = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['n'] = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    phoneme_to_feature['m'] = [1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['ŋ'] = [1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['ɲ'] = [1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['s'] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    phoneme_to_feature['z'] = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]

    # turkish sh
    phoneme_to_feature['ş'] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    phoneme_to_feature['ʃ'] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    # turkish jeep

    phoneme_to_feature['c'] = [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0.5, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

    # turkish cheap

    phoneme_to_feature['ç'] = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.5, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

    # spanish
    phoneme_to_feature['ʧ'] = [1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0.5, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]

    phoneme_to_feature['f'] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    phoneme_to_feature['v'] = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

    phoneme_to_feature['r'] = [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    phoneme_to_feature['l'] = [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    phoneme_to_feature['w'] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0]

    phoneme_to_feature['j'] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # change into y for turkish [j]
    phoneme_to_feature['y'] = [0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # turkish j is meaSure
    phoneme_to_feature['J'] = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    phoneme_to_feature['h'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['a'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['o'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0]
    phoneme_to_feature['u'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0]

    phoneme_to_feature['ä'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    phoneme_to_feature['ö'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    phoneme_to_feature['ü'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]

    # finnish y
    phoneme_to_feature['Y'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0]

    phoneme_to_feature['e'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    phoneme_to_feature['i'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    # turkish ɯ

    phoneme_to_feature['ı'] = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]

    # turkish ğ

    phoneme_to_feature['ğ'] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # spanish

    phoneme_to_feature['ɾ'] = [1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0]

    phoneme_to_feature['B'] = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['D'] = [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    phoneme_to_feature['θ'] = [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]

    phoneme_to_feature['G'] = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    phoneme_to_feature['ʎ'] = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

    # spanish j
    phoneme_to_feature['K'] = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]

    return phoneme_to_feature


def transform(sequence, language):
    """
    Transcribes sequence into phonetic representation
    :param sequence: sequence to transcribe
    :param language: language whose phonetic rules are used
    :return: transcribed sequence
    """
    assert language in ["FI", "ES", "TUR"]

    if language == 'FI':
        sequence = sequence.replace('y', 'Y')
    elif language == 'ES':
        sequence = sequence.replace('j', 'K')
        sequence = sequence.replace('ll', 'ʎ')
        sequence = sequence.replace('r', 'ɾ')
        sequence = sequence.replace('c', 'θ')
        sequence = sequence.replace('ñ', 'ɲ')
        sequence = sequence.replace('qu', 'k')
        sequence = sequence.replace('ó', 'o')
        sequence = sequence.replace('á', 'a')
        sequence = sequence.replace('í', 'i')
        sequence = sequence.replace('x', 'ks')  # FIXME!
        sequence = [c for c in sequence]
        for i, c in enumerate(sequence):
            if i > 0 and sequence[i - 1] in VOWELS:
                if c in SPREPLACE:
                    sequence[i] = SPREPLACE[c]
        sequence = ''.join(sequence)
    elif language == "TUR":
        sequence = sequence.replace('j', 'J')
    return sequence

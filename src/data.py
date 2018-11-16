from collections import Counter

from src.features import transform

# Indices for wordform, lemma and tags in the data list.
WF = 1
LEMMA = 2
TAGS = 3

# Minimum number of character occurrences we require for embedded
# characters.
MINCHAROCC = 100

# Word boundary.
WB = '#'


def read_data(filename, language):
    """


    :param filename: file in src.data
    :param language: FI, ES, TUR
    :return: data: list of encoded (word_form, lemma, tags), character_encoder: character encoder dict ,
        tags: void dictionary, embedded_chars: set of embedded characters
    """
    data = []
    char_counts = Counter()

    # We need to first count all characters to make sure that
    # characters occurring more than MINCHAROCC will get the first 0
    # ... N character codes.
    with open(filename, "r") as f:
        for line in f:
            word_form, tags, lemma = line.strip("\n").lower().split('\t')
            word_form = transform(word_form, language)
            lemma = transform(lemma, language)
            char_counts.update(word_form)

            # count(word_form, charcounts)
            data.append((word_form, lemma, tags))

    char_counts = sorted([(_count, char) for char, _count in char_counts.items()],
                         reverse=True)
    # TODO why does tag_encoder remain void?
    tag_encoder = {}

    character_encoder = {x[1]: i for i, x in enumerate(char_counts)}

    for i, d in enumerate(data):
        word_form, lemma, tags = d
        word_form = [character_encoder.setdefault(c, len(character_encoder)) for c in WB + word_form + WB]
        tags = [tag_encoder.setdefault(t, len(tag_encoder)) for t in tags.split(',')]
        lemma = [character_encoder.setdefault(c, len(character_encoder)) for c in WB + lemma + WB]
        data[i] = (word_form, lemma, tags)

    embedded_chars = set([character_encoder[c] for _count, c in char_counts if _count > MINCHAROCC])
    return data, character_encoder, tag_encoder, embedded_chars

# Rory Linerud
# Unlabeled Dependency Parser

from conllu import parse_incr, TokenList
import pickle

_feature_weights = {}


def get_arcs(tokens: TokenList) -> set[tuple[int, int]]:
    """Return the set of arcs that comprise the specified TokenList."""
    return {(token['head'], token['id']) for token in tokens if token['head'] is not None}


def get_features(arc: tuple[int, int], sentence: list[tuple[str, str, str]]) -> set[str]:
    """Return the set of features that the specified arc exhibits."""
    head_form, head_lemma, head_tag = sentence[arc[0]]
    dep_form, dep_lemma, dep_tag = sentence[arc[1]]
    direction = 'L' if arc[0] - arc[1] > 0 else 'R'
    length = abs(arc[0] - arc[1])

    head_neighbor = sentence[arc[0] - 1] if direction == 'L' else sentence[arc[0] + 1]
    dep_neighbor = sentence[arc[1] + 1] if direction == 'L' else sentence[arc[1] - 1]

    features = {
        f'HF_{head_form}',
        f'HF/DF_{head_form}/{dep_form}',
        f'HF/DF/D_{head_form}/{dep_form}/{direction}',
        f'HF/DF/L_{head_form}/{dep_form}/{length}',
        f'HF/DF/D/L_{head_form}/{dep_form}/{direction}/{length}',
        f'HL_{head_lemma}',
        f'HL/DL_{head_lemma}/{dep_lemma}',
        f'HL/DL/D_{head_lemma}/{dep_lemma}/{direction}',
        f'HL/DL/L_{head_lemma}/{dep_lemma}/{length}',
        f'HL/DL/D/L_{head_lemma}/{dep_lemma}/{direction}/{length}',
        f'HT_{head_tag}',
        f'HT/DT_{head_tag}/{dep_tag}',
        f'HT/DT/D_{head_tag}/{dep_tag}/{direction}',
        f'HT/DT/L_{head_tag}/{dep_tag}/{length}',
        f'HT/DT/D/L_{head_tag}/{dep_tag}/{direction}/{length}',
        f'HF/HT_{head_form}/{head_tag}',
        f'D_{direction}',
        f'L_{length}',
        f'HNF_{head_neighbor[0]}',
        f'HF/HNF_{head_form}/{head_neighbor[0]}',
        f'HNL_{head_neighbor[1]}',
        f'HL/HNL_{head_lemma}/{head_neighbor[1]}',
        f'HNT_{head_neighbor[2]}',
        f'HT/HNT_{head_tag}/{head_neighbor[2]}',
        f'DNF_{dep_neighbor[0]}',
        f'DF/DNF_{dep_form}/{dep_neighbor[0]}',
        f'DNL_{dep_neighbor[1]}',
        f'DL/DNL_{dep_lemma}/{dep_neighbor[1]}',
        f'DNT_{dep_neighbor[2]}',
        f'DT/DNT_{dep_tag}/{dep_neighbor[2]}'
    }

    return features


def update_weights(arc: tuple[int, int], sentence: list[tuple[str, str, str]], step: float) -> None:
    """Update the weights vector according to the features present in the specified arc."""
    for feature in get_features(arc, sentence):
        if feature in _feature_weights:
            _feature_weights[feature] += step
        else:
            _feature_weights[feature] = step


def score_arc(arc: tuple[int, int], sentence: list[tuple[str, str, str]]) -> float:
    """Return the score associated with the specified arc."""
    return sum([_feature_weights[feature] for feature in get_features(arc, sentence) if feature in _feature_weights])


def learn_weights(data_file: str) -> None:
    """Train on the specified data set, saving the weights vector to file after every iteration."""
    iteration = 0
    while True:
        with open(data_file) as data:
            for token_list in parse_incr(data):
                sentence = get_sentence(token_list)
                parse = eisner_parse(sentence)
                gold = get_arcs(token_list)
                if not parse == gold:
                    for arc in gold:
                        update_weights(arc, sentence, 1.0)
                    for arc in parse:
                        update_weights(arc, sentence, -1.0)

            save_weights(iteration)
            iteration += 1


def save_weights(iteration: int) -> None:
    """Save the weights vector to file."""
    with open(f'weights{iteration}.pkl', 'wb') as weights:
        pickle.dump(_feature_weights, weights, pickle.HIGHEST_PROTOCOL)
        print(f'weights{iteration}.pkl saved')


def load_weights(weights_file: str):
    """Load the weights vector from file."""
    with open(weights_file, 'rb') as weights:
        global _feature_weights
        _feature_weights = pickle.load(weights)


def test_weights(data_file: str) -> None:
    """Test the weights vector on the specified set of test data, reporting its performance."""
    with open(data_file) as data:
        scores = []
        for token_list in parse_incr(data):
            sentence = get_sentence(token_list)
            parse = eisner_parse(sentence)
            gold = get_arcs(token_list)
            score = sum([1.0 for arc in parse if arc in gold]) / len(parse)
            scores.append(score)

        print(f'Recorded accuracy: {sum(scores) / len(scores)}')


def eisner_parse(sentence: list[tuple[str, str, str]]) -> set[tuple[int, int]]:
    """Return the highest scoring dependency parse for the given input sentence."""
    scores, arcs = {}, {}
    for i in range(len(sentence)):
        for d in {'L', 'R'}:
            for s in {'C', 'I'}:
                scores[(i, i, d, s)] = 0.0
                arcs[(i, i, d, s)] = set()

    for span in range(1, len(sentence)):
        for i in range(len(sentence)):
            if (j := i + span) >= len(sentence):
                break
            else:
                scores[(i, j, 'L', 'I')], argmax = max(
                    [(scores[(i, k, 'R', 'C')] +
                     scores[(k + 1, j, 'L', 'C')] +
                     score_arc((j, i), sentence), k) for k in range(i, j)], key=lambda x: x[0])
                arcs[(i, j, 'L', 'I')] = arcs[(i, argmax, 'R', 'C')] | arcs[(argmax + 1, j, 'L', 'C')] | {(j, i)}

                scores[(i, j, 'R', 'I')], argmax = max(
                    [(scores[(i, k, 'R', 'C')] +
                     scores[(k + 1, j, 'L', 'C')] +
                     score_arc((i, j), sentence), k) for k in range(i, j)], key=lambda x: x[0])
                arcs[(i, j, 'R', 'I')] = arcs[(i, argmax, 'R', 'C')] | arcs[(argmax + 1, j, 'L', 'C')] | {(i, j)}

                scores[(i, j, 'L', 'C')], argmax = max(
                    [(scores[(i, k, 'L', 'C')] +
                     scores[(k, j, 'L', 'I')], k) for k in range(i, j)], key=lambda x: x[0])
                arcs[(i, j, 'L', 'C')] = arcs[(i, argmax, 'L', 'C')] | arcs[(argmax, j, 'L', 'I')]

                scores[(i, j, 'R', 'C')], argmax = max(
                    [(scores[(i, k, 'R', 'I')] +
                     scores[(k, j, 'R', 'C')], k) for k in range(i + 1, j + 1)], key=lambda x: x[0])
                arcs[(i, j, 'R', 'C')] = arcs[(i, argmax, 'R', 'I')] | arcs[(argmax, j, 'R', 'C')]

    return arcs[(0, len(sentence) - 1, 'R', 'C')]


def get_sentence(tokens: TokenList) -> list[tuple[str, str, str]]:
    """Return the word forms, lemmas, and POS tags associated with the specified TokenList's sentence."""
    return [('__root__', '__root__', '__root__')] + [(token['form'], token['lemma'], token['xpos']) for token in tokens]


if __name__ == '__main__':
    learn_weights('./Universal Dependencies 2.7/ud-treebanks-v2.7/UD_German-HDT/de_hdt-ud-train.conllu')
    #learn_weights('./Universal Dependencies 2.7/ud-treebanks-v2.7/UD_English-GUM/en_gum-ud-train.conllu')
    #learn_weights('./Universal Dependencies 2.7/ud-treebanks-v2.7/UD_English-EWT/en_ewt-ud-train.conllu')

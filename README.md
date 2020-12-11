# parseptron
A Python implementation of an unlabeled dependency parsing system. Uses
Eisner's dependency parsing algorithm to find the best parse tree for a
sentence given a set of features and their corresponding weights, and uses
the structured perceptron algorithm to learn these feature weights. Features
involve not just the various word forms that are present in the input sentence,
but also their corresponding lemmas and part of speech tags.

As it is currently written, parseptron can only train on and parse sentences
that come from .conllu-formatted Universal Dependencies files.

(Usage examples and examples of JSON-formatted weights files still yet to come.)

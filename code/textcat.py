#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.
"""
import argparse
import logging
import math
from pathlib import Path
import numpy
import matplotlib.pyplot as plt

from probs import LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model1",
        type=Path,
        help="path to the first trained model",
    )
    parser.add_argument(
        "model2",
        type=Path,
        help="path to the second trained model",
    )
    parser.add_argument(
        "prior_prob",
        type=float,
        help="prior probability of first category",
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="*"
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v",
        "--verbose",
        action="store_const",
        const=logging.DEBUG,
        default=logging.INFO,
    )
    verbosity.add_argument(
        "-q", "--quiet", dest="verbose", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel, prior_prob: float) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z) * prior_prob  # p(z | xy)
        log_prob += math.log(prob)
    return log_prob


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm1 = LanguageModel.load(args.model1)
    lm2 = LanguageModel.load(args.model2)
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    if (lm2.vocab != lm1.vocab):
        log.error("Models need the same vocab file")
    log.info("Per-file log-probabilities:")
    total_log_prob1 = 0.0
    total_log_prob2 = 0.0
    cat1_name = args.model1
    cat2_name = args.model2
    cat1_count = 0
    cat2_count = 0
    total_count = 0

    total_log_prob = 0.0
    num_words_list = []
    classification_accuracy_list = []

    for file in args.test_files:
        classification_accuracy = 0
        log_prob1: float = file_log_prob(file, lm1, args.prior_prob)
        log_prob2: float = file_log_prob(file, lm2, (1 - args.prior_prob))
        #print(f"{log_prob:g}\t{file}")
        if log_prob1 > log_prob2:
            print(cat1_name,' ', file) #generalize for other category names
            cat1_count += 1
        else:
            print(cat2_name, ' ', file)
            cat2_count += 1
            classification_accuracy = 1

        file_name = file.stem
        file_num_words = file_name.split('.')[1]
        num_words_list.append(float(file_num_words))
        classification_accuracy_list.append(classification_accuracy)
        total_count += 1
        total_log_prob1 += log_prob1
        total_log_prob2 += log_prob2

    print(cat1_count, "files were more probably", cat1_name, "(" + str(cat1_count / total_count) + "%)")
    print(cat2_count, "files were more probably", cat2_name, "(" + str(cat2_count / total_count) + "%)")
    bits1 = -total_log_prob1 / math.log(2)   # convert to bits of surprisal
    bits2 = -total_log_prob2 / math.log(2)   # convert to bits of surprisal
    tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    #print(f"Overall cross-entropy for first model:\t{bits1 / tokens:.5f} bits per token")
    #print(f"Overall cross-entropy for second model:\t{bits2 / tokens:.5f} bits per token")

    '''correlation_matrix = numpy.corrcoef(num_words_list, classification_accuracy_list)
    correlation_xy = correlation_matrix[0, 1]
    r_squared = correlation_xy ** 2
    print('Correlation =', r_squared)

    plt.plot(num_words_list, classification_accuracy_list, 'rx')
    plt.xlabel("Number of words in file")
    plt.ylabel("Classified Correctly (1) or Incorrectly (0)")
    plt.xlim(left=-0, right=700)
    plt.title("Spam Files Number of Words vs Classification Accuracy")
    plt.show()'''


if __name__ == "__main__":
    main()
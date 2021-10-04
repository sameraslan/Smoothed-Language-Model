#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.
"""
import argparse
import logging
import math
from pathlib import Path

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


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    for (x, y, z) in read_trigrams(file, lm.vocab):
        prob = lm.prob(x, y, z)  # p(z | xy)
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

    log.info("Per-file log-probabilities:")
    total_log_prob1 = 0.0
    total_log_prob2 = 0.0
    cat1_count = 0
    cat2_count = 0
    total_count = 0

    for file in args.test_files:
        log_prob1: float = file_log_prob(file, lm1)
        log_prob2: float = file_log_prob(file, lm2)
        #print(f"{log_prob:g}\t{file}")
        if log_prob1 > log_prob2:
            print('gen',' ', file) #generalize for other category names
            cat1_count += 1
        else:
            print('spam', ' ', file) #generalize for other category names
            cat2_count += 1
        total_count += 1
        total_log_prob1 += log_prob1
        total_log_prob2 += log_prob2

    print(cat1_count, "files were more probably", "gen (" + str(cat1_count / total_count) + "%)") #generalize for other category names
    print(cat2_count, "files were more probably", "spam (" + str(cat2_count / total_count) + "%)") #generalize for other category names
    bits1 = -total_log_prob1 / math.log(2)   # convert to bits of surprisal
    bits2 = -total_log_prob2 / math.log(2)   # convert to bits of surprisal
    tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    #print(f"Overall cross-entropy for first model:\t{bits1 / tokens:.5f} bits per token")
    #print(f"Overall cross-entropy for second model:\t{bits2 / tokens:.5f} bits per token")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import matplotlib.pyplot as plt

from probs import LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # Basically the only okay global variable.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model",
        type=Path,
        help="path to the trained model",
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
    lm = LanguageModel.load(args.model)
    # We use natural log for our internal computations and that's
    # the kind of log-probability that file_log_prob returns.
    # But we'd like to print a value in bits: so we convert
    # log base e to log base 2 at print time, by dividing by log(2).

    log.info("Per-file log-probabilities:")
    total_log_prob = 0.0
    num_words_list = []
    total_probs_list = []
    for file in args.test_files:
        file_name = file.stem
        file_num_words = file_name.split('.')[1]
        num_words_list.append(float(file_num_words))
        log_prob: float = file_log_prob(file, lm)
        total_probs_list.append(float(log_prob)/float(file_num_words))
        print(f"{log_prob:g}\t{file}")
        total_log_prob += log_prob

    #devbits = 87482
    bits = -total_log_prob / math.log(2)   # convert to bits of surprisal
    tokens = sum(num_tokens(test_file) for test_file in args.test_files)
    print(f"Overall cross-entropy:\t{bits / tokens:.5f} bits per token")

    plt.plot(num_words_list, total_probs_list, 'rx')
    plt.xlabel("Number of words in file")
    plt.ylabel("Lob Probability / Number of words")
    plt.title("Gen Files Number of Words vs Log Probability")
    plt.show()

    #print(f"Dev cross entropy:\t{bits / devbits:.5f} bits per token") #used to calculate dev bits for 3e


if __name__ == "__main__":
    main()

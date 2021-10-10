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
        "model",
        type=Path,
        help="path to the trained model",
    )
    parser.add_argument(
        "samples",
        type=int,
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="max length of each sample"
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


def lm_sample(lm: LanguageModel, max_length: int) -> str:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """

    (x, y) = ("BOS", "BOS")
    z = lm.sample(x, y, lm.vocab)
    (x, y) = (y, z)

    sentence = z
    length = 1

    while z != "EOS" and length < max_length:
        length += 1

        z = lm.sample(x, y, lm.vocab)
        (x, y) = (y, z)

        sentence += " " + z

    sentence += "."

    return sentence


def main():
    args = parse_args()
    logging.basicConfig(level=args.verbose)

    log.info("Testing...")
    lm = LanguageModel.load(args.model)
    max_length = args.max_length

    for i in range(args.samples):
        sentence = lm_sample(lm, max_length)
        print(sentence)


if __name__ == "__main__":
    main()

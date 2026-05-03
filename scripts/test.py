"""Inference / test entry point required by the hackathon submission spec.

The hackathon documentation (section 3.i) mandates that the submission
packaged folder include a `test.py` script. This file is a thin wrapper
around `scripts/predict.py`, which carries the full implementation
(HFlip TTA, optional ensemble, raw_id / class_id / color output).

Run `python scripts/test.py --help` for the full argument list.
"""

from predict import main

if __name__ == "__main__":
    main()

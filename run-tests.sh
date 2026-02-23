#!/bin/bash

source .venv/bin/activate

export PYTHONPATH=.

behave linalg/features/
pytest test_linalg_pytest/test_vector.py -v
pytest test_linalg_pytest/test_matrix.py -v
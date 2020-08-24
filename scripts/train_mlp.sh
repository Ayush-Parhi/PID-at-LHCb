#!/bin/bash
pipenv run python training/run_experiment.py --save '{"model": "MlpModel", "network": "mlp", "train_args": {"batch_size": 1024}}'
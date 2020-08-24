#!/bin/bash
pipenv run python training/run_experiment.py --save '{"model": "CnnModel", "network": "cnn", "train_args": {"batch_size": 1024}}'
#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = football-analytics
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python Dependencies
.PHONY: uv
uv:
	uv venv
	. .venv/bin/activate && uv pip install -e .

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff and isort
.PHONY: lint
lint:
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff format

## Run mypy
.PHONY: mypy
mypy:
	mypy .

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests

## Run tests with coverage
.PHONY: coverage
coverage:
	$(PYTHON_INTERPRETER) -m pytest --cov=football_analytics --cov-report=term-missing tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: 
	$(PYTHON_INTERPRETER) ai/dataset.py


## Make train
# Usage example: make train training_config_path=configurations/train.json
.PHONY: train $(training_config_path)
train:
	$(PYTHON_INTERPRETER) ai/modeling/train.py $(training_config_path)


## Make validate
# Usage example: make validate validation_config_path=configurations/validate.json
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) ai/modeling/validate.py $(validation_config_path)


#################################################################################
# EXPERIMENTS                                                                   #
#################################################################################


## Make optuna hyperparameter search experiment
# Usage example: make optuna_hyperparameter_search path_to_hyperparameters_search_config=configurations/hyperparameter_search.json
.PHONY: optuna_hyperparameter_search $(path_to_hyperparameters_search_config)
optuna_hyperparameter_search:
	$(PYTHON_INTERPRETER) ai/experiments/yolo11_optuna_hyperparameter_search.py $(path_to_hyperparameters_search_config)


## Make tensorboard
# Use example: make tensorboard path_to_logs=./runs
# Use example: make tensorboard path_to_logs=./football_analytics/experiments/runs
.PHONY: tensorboard $(path_to_logs)
tensorboard:
	tensorboard --logdir $(path_to_logs)


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)

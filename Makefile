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
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt


## Install Python Dependencies with uv
.PHONY: uv
uv:
	uv sync
	uv pip install -e .


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 football_analytics
	isort --check --diff --profile black football_analytics
	black --check --config pyproject.toml football_analytics

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml football_analytics




## Set up python interpreter environment
.PHONY: create_environment
create_environment:
	@rm -rf .venv
	$(PYTHON_INTERPRETER)$(PYTHON_VERSION) -m venv .venv
	@echo ">>> New python interpreter environment created. Activate it using 'source .venv/bin/activate'"


.PHONY: freeze
freeze:
	$(PYTHON_INTERPRETER) -m pip freeze > requirements.txt


## Run the service
.PHONY: run_service
run_service:
	uvicorn services.track.app:app --host 0.0.0.0 --port 8000 --reload

## Run players detection
# Usage example:
# make run_players_detection source_video_path=data/input/test_video.mp4 output_video_path=data/output/output_video.mp4
.PHONY: run_analytics $(source_video_path) $(output_video_path)
run_analytics:
	$(PYTHON_INTERPRETER) scripts/run_analytics.py $(source_video_path) $(output_video_path)


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
# Usage example: make validate models_to_validate="runs/detect/train2/weights/best.pt runs/detect/train3/weights/best.pt"
.PHONY: validate
models_to_validate ?=
validate:
	$(PYTHON_INTERPRETER) ai/modeling/validate.py $(models_to_validate)


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

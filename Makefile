#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = football-analytics
PYTHON_VERSION = 3.12
PYTHON_INTERPRETER = python

#################################################################################
# SERVICES                                                                      #
#################################################################################


.PHONY: dev-services
dev-services:
	@echo "Starting ball_inference_service, player_inference_service, and keypoints_detection_service..."
	@tmux new-session -d -s ball-inference-service 'make run-ball-inference-service'
	@tmux new-session -d -s player-inference-service 'make run-player-inference-service'
	@tmux new-session -d -s keypoints-detection-service 'make run-keypoints-detection-service'
	@echo "Services started."

.PHONY: full-stack
full-stack: dev-services
	@echo "Starting inference_manager_service..."
	@tmux new-session -d -s inference-manager-service 'make run-inference-manager-service'
	@echo "Service started. Full stack is running."

.PHONY: run-ball-inference-service
run-ball-inference-service:
	$(PYTHON_INTERPRETER) services/ball_inference/ball_inference_service.py

.PHONY: run-player-inference-service
run-player-inference-service:
	$(PYTHON_INTERPRETER) services/player_inference/player_inference_service.py

.PHONY: run-keypoints-detection-service
run-keypoints-detection-service:
	$(PYTHON_INTERPRETER) services/keypoints_detection/keypoints_detection_service.py

.PHONY: run-inference-manager-service
run-inference-manager-service:
	$(PYTHON_INTERPRETER) services/inference_manager/inference_manager_service.py

.PHONY: stop-services
stop-services:
	@echo "Stopping all services..."
	@ps aux | grep "services/" | grep -v grep | awk '{print $$2}' | xargs kill -9 || true
	@echo "All services stopped."



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

## Download models from kaggle
.PHONY: download-models
download_models:
	$(PYTHON_INTERPRETER) scripts/download_models.py


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
# Usage example: make optuna-hyperparameter-search path_to_hyperparameters_search_config=configurations/hyperparameter_search.json
.PHONY: optuna-hyperparameter-search $(path_to_hyperparameters_search_config)
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

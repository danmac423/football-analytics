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
# Usage example: make run_players_detection source_video_path=video.mp4 output_video_path=output.mp4
.PHONY: run_players_detection $(source_video_path) $(output_video_path)
run_players_detection:
	$(PYTHON_INTERPRETER) core/annotations.py $(source_video_path) $(output_video_path)


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) football_analytics/dataset.py


## Make train
# Usage example: make train training_config_path=../configurations/train.json
.PHONY: train $(training_config_path)
train:
	cd models && $(PYTHON_INTERPRETER) ../football_analytics/modeling/train.py $(training_config_path)



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

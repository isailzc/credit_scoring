#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = credit_scoring
PYTHON_VERSION = 3.9
PYTHON_INTERPRETER = uv run python

# Rutas clave (Testigos de ejecución del pipeline)
RAW_DATA = data/raw/credit_scoring.csv
VALIDATION_DATA = data/raw/validacion.csv
PROCESSED_DATA_FLAG = data/processed/x_pd.pkl
MODELS_FLAG = models/pipe_ejecucion_pd.pickle
PREDICTIONS_FILE = data/processed/predictions.csv

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	uv sync

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES (PIPELINE)                                                      #
#################################################################################

## Ejecutar el pipeline completo (Procesar -> Entrenar -> Predecir)
.PHONY: pipeline
pipeline: predict

## 1. Make dataset: Procesa datos crudos y genera matrices
.PHONY: data
data: $(PROCESSED_DATA_FLAG)

$(PROCESSED_DATA_FLAG): $(RAW_DATA)
	@echo ">>> Procesando datos crudos..."
	$(PYTHON_INTERPRETER) credit_scoring/dataset.py

## 2. Entrenar modelos: PD, EAD y LGD
.PHONY: train
train: $(MODELS_FLAG)

$(MODELS_FLAG): $(PROCESSED_DATA_FLAG)
	@echo ">>> Entrenando modelos..."
	$(PYTHON_INTERPRETER) credit_scoring/modeling/train.py

## 3. Generar predicciones y Pérdida Esperada
.PHONY: predict
predict: $(PREDICTIONS_FILE)

$(PREDICTIONS_FILE): $(MODELS_FLAG) $(VALIDATION_DATA)
	@echo ">>> Generando predicciones..."
	$(PYTHON_INTERPRETER) credit_scoring/modeling/predict.py

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
	@uv run python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
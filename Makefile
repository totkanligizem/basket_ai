SHELL := /bin/bash
PYTHON ?= python

.PHONY: help install pipeline pipeline-fast train export-dashboard dashboard checks

help:
	@echo "Targets:"
	@echo "  make install           - Install Python dependencies"
	@echo "  make pipeline          - Run full production pipeline"
	@echo "  make pipeline-fast     - Run pipeline without embeddings retrain"
	@echo "  make train             - Run leakage-safe ranker training"
	@echo "  make export-dashboard  - Export dashboard CSV artifacts"
	@echo "  make dashboard         - Start Dash app"
	@echo "  make checks            - Compile/smoke checks"

install:
	$(PYTHON) -m pip install -r requirements.txt

pipeline:
	PYTHON_BIN=$(PYTHON) ./scripts/run_production_pipeline.sh

pipeline-fast:
	PYTHON_BIN=$(PYTHON) ./scripts/run_production_pipeline.sh --skip-embeddings

train:
	$(PYTHON) scripts/train_ranker_leakage_safe.py

export-dashboard:
	$(PYTHON) scripts/export_dashboard_artifacts.py

dashboard:
	$(PYTHON) dash_app/app.py

checks:
	$(PYTHON) -m compileall src scripts dash_app/app.py

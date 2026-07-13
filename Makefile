.DEFAULT_GOAL := help

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

DFINE_DIR := external/D-FINE
DFINE_REPO := https://github.com/Peterande/D-FINE.git
DFINE_REQ := $(DFINE_DIR)/requirements.txt
PY_STAMP := $(VENV)/.deps-installed

WEIGHTS := weights/dfine_n_coco.pth
WEIGHTS_URL := https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth

NODE_STAMP := node_modules/.package-lock.json

INTERVIEW_REQ := requirements-interview.txt
INTERVIEW_STAMP := $(VENV)/.interview-deps-installed
INTERVIEW_MODELS_STAMP := $(VENV)/.interview-models-downloaded

.PHONY: help setup venv dfine weights node-deps py-deps install start agent dev watch clean distclean interview-deps interview-models interview

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup: venv dfine weights node-deps py-deps ## Bootstrap a fresh clone from scratch (idempotent, safe to re-run)

venv: $(PYTHON) ## Create the Python virtualenv if missing

$(PYTHON):
	python3 -m venv $(VENV)

dfine: $(DFINE_DIR)/.git ## Clone the D-FINE source if missing

$(DFINE_DIR)/.git:
	mkdir -p external
	git clone $(DFINE_REPO) $(DFINE_DIR)

weights: $(WEIGHTS) ## Download the D-FINE-N COCO checkpoint if missing

$(WEIGHTS):
	mkdir -p weights
	curl -L -o $(WEIGHTS) $(WEIGHTS_URL)

node-deps: $(NODE_STAMP) ## Install npm dependencies if missing or stale

$(NODE_STAMP): package.json package-lock.json
	npm install

py-deps: $(PY_STAMP) ## Install Python dependencies into the venv if missing or stale

$(PY_STAMP): $(PYTHON) $(DFINE_REQ)
	$(PIP) install -r $(DFINE_REQ)
	touch $(PY_STAMP)

install: setup ## Alias for setup

start: node-deps ## Run the LiveKit web server (server.js)
	npm start

agent: node-deps py-deps weights ## Run the LiveKit vision agent (agent.js)
	npm run agent

dev: node-deps py-deps weights ## Run the server and agent together
	npm start & npm run agent & wait

interview-deps: $(INTERVIEW_STAMP) ## Install the voice interview agent's Python dependencies if missing or stale

$(INTERVIEW_STAMP): $(PYTHON) $(INTERVIEW_REQ)
	$(PIP) install -r $(INTERVIEW_REQ)
	touch $(INTERVIEW_STAMP)

interview-models: $(INTERVIEW_MODELS_STAMP) ## Pre-fetch the turn-detector model weights if missing

$(INTERVIEW_MODELS_STAMP): $(INTERVIEW_STAMP)
	$(PYTHON) -m livekit.agents download-files
	touch $(INTERVIEW_MODELS_STAMP)

interview: node-deps interview-deps interview-models ## Run the LiveKit voice interview agent (interview_agent.py)
	$(PYTHON) interview_agent.py dev

watch: py-deps weights ## Run the standalone person-watch camera monitor
	$(PYTHON) person_watch.py --source 0

clean: ## Remove caches and bytecode (keeps venv, node_modules, weights)
	rm -rf __pycache__ .cache
	find . -name '*.pyc' -not -path './node_modules/*' -not -path './.venv/*' -delete

distclean: clean ## Remove venv, node_modules, cloned D-FINE source and downloaded weights
	rm -rf $(VENV) node_modules $(DFINE_DIR) $(WEIGHTS)

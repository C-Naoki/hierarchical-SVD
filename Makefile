.PHONY: install
install:
	@if ! command -v pyenv &> /dev/null; then \
		echo "Error: pyenv is not installed. Visit https://github.com/pyenv/pyenv#installation for installation instructions."; \
		exit 1; \
	fi
	@if ! command -v poetry &> /dev/null; then \
		echo "Error: poetry is not installed. Visit https://python-poetry.org/docs/#installation for installation instructions."; \
		exit 1; \
	fi
	@if [ -z "$$(pyenv versions | grep '3\.10\..*')" ]; then \
		pyenv install 3.10; \
	else \
		echo "Python 3.10 is already installed."; \
	fi
	poetry env use 3.10
	poetry install
	poetry run pre-commit install
	@VENV_PATH=$$(poetry env info --path); \
	SITE_PACKAGES_DIR="$${VENV_PATH}/lib/python3.10/site-packages"; \
	PROJECT_ROOT=$$(echo $${VENV_PATH} | rev | cut -d'/' -f2- | rev); \
	VENV_NAME=$$(basename `dirname $${VENV_PATH}`); \
	echo $${PROJECT_ROOT} > $${SITE_PACKAGES_DIR}/$${VENV_NAME}.pth

.PHONY: run
run:
	python src/main.py

.PHONY: pre-commit
pre-commit:
	poetry run pre-commit run --all-files

.PHONY: test
test:
	poetry run pytest -s -vv --cov=. --cov-branch --cov-report=html

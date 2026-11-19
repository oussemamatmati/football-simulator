.PHONY: install data train evaluate app test lint all

install:
	pip install -e ".[app,dev]"

data:
	footballsim download

train:
	footballsim train

evaluate:
	footballsim evaluate

app:
	footballsim app

test:
	pytest -q

lint:
	ruff check src tests app

all: data train evaluate

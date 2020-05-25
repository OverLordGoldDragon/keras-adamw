#!/usr/bin/env bash
pycodestyle --max-line-length=89 keras_adamw tests

pytest -s --cov=keras_adamw tests/

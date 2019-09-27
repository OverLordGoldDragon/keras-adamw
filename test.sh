#!/usr/bin/env bash
pycodestyle --max-line-length=89 keras_adamw tests && \
    nosetests --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --ignore-files="example.py" \
    --cover-package=keras_adamw --with-doctest tests

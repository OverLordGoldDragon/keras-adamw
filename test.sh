#!/usr/bin/env bash
pycodestyle --max-line-length=89 keras_adamw tests


if [[ "$TF_VERSION" == "1.14.0" ]] && [[ "$KERAS_VERSION" == "2.2.5" ]]; then
    if [[ "$TF_KERAS" == "True" ]]; then
        TESTDIR=tests/test_optimizers_225tf
    else
        TESTDIR=tests/test_optimizers_225
    fi
elif [[ "$TF_VERSION" == "2.0.0" ]] && [[ "$KERAS_VERSION" == "2.3.0" ]]; then
    if [[ "$TF_KERAS" == "True" ]]; then
        TESTDIR=tests/test_optimizers_v2
    else
	TESTDIR=tests/test_optimizers
    fi
fi

nosetests \
    --nocapture --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov \
	--with-doctest "$TESTDIR"

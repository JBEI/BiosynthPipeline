#!/bin/sh
set -e -x

# Run all tests and calculate coverage.
coverage run --source=. -m pytest
coverage html
coverage report

# Upload code coverage to CodeCov.
bash < (curl -s https://codecov.io/bash)

#!/bin/sh
# Run from the repository root in Linux x86_64 with CPython 3.11.
set -eu
python -c 'import platform, sys; assert sys.version_info[:2] == (3, 11) and sys.platform == "linux" and platform.machine() == "x86_64", "Locks target Linux x86_64 / Python 3.11"'
python -m pip install --disable-pip-version-check 'pip==25.3' 'pip-tools==7.6.1' 'setuptools==79.0.1' wheel
export CUSTOM_COMPILE_COMMAND='sh scripts/compile-locks.sh'
pip-compile --quiet --resolver=backtracking --generate-hashes --allow-unsafe --strip-extras \
  --output-file=requirements/runtime.lock requirements/runtime.in ml/pyproject.toml "$@"
pip-compile --quiet --resolver=backtracking --generate-hashes --allow-unsafe --strip-extras --extra=dev \
  --output-file=requirements/dev.lock requirements/dev.in ml/pyproject.toml "$@"

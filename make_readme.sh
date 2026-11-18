#!/usr/bin/env bash
# Regenerate README.md from demo.ipynb.
# Outputs are rendered as plain text instead of HTML, because PyPI does not
# render the HTML tables pandas produces for DataFrames.
set -euo pipefail

jupyter nbconvert demo.ipynb --to markdown --output README \
  --NbConvertBase.display_data_priority="['text/markdown','image/png','text/plain']"

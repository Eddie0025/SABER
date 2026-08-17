#!/bin/bash
# SABER — Main Entrypoint
# Usage: ./run.sh

set -e

echo "Starting SABER..."
python3 chat.py "$@"

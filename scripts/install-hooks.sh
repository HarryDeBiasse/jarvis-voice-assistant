#!/usr/bin/env bash
# Activate the repo's secret-scanning pre-commit hook.
set -e
git config core.hooksPath .githooks
echo "Hooks installed: commits are now scanned for private IPs / MACs / keys."

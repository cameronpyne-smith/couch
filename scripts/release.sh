#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(uv version --short)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "pyproject.toml version is not X.Y.Z: $version" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
git fetch --tags --quiet origin
! git rev-parse -q --verify "refs/tags/v$version" >/dev/null || { echo "v$version already released; bump pyproject.toml" >&2; exit 1; }
image=ghcr.io/cameronpyne-smith/couch
docker buildx build --push -t "$image:$version" -t "$image:latest" .
git tag "v$version"
git push origin HEAD "v$version"
echo "$image:$version"

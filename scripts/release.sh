#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(uv version --short)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "pyproject.toml version is not X.Y.Z: $version" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
[ -z "$(git ls-remote --tags origin "refs/tags/v$version")" ] || { echo "v$version already released; bump pyproject.toml" >&2; exit 1; }
image=ghcr.io/cameronpyne-smith/couch
docker buildx build --push -t "$image:$version" -t "$image:latest" .
docker buildx build --push -f serve/Dockerfile -t "$image-server:$version" -t "$image-server:latest" .
git tag "v$version" 2>/dev/null || [ "$(git rev-parse "v$version^{}")" = "$(git rev-parse HEAD)" ] || { echo "local tag v$version is on another commit" >&2; exit 1; }
git push origin HEAD "v$version"
echo "$image:$version"
echo "$image-server:$version"

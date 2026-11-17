#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
version=$(uv version --short)
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "pyproject.toml version is not X.Y.Z: $version" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "working tree not clean" >&2; exit 1; }
if [[ $(git remote get-url origin) != http* ]]; then
    agent=0
    ssh-add -l >/dev/null 2>&1 || agent=$?
    if [ "$agent" -eq 2 ]; then
        eval "$(ssh-agent -s)" >/dev/null
        trap 'ssh-agent -k >/dev/null' EXIT
    fi
    [ "$agent" -eq 0 ] || ssh-add
fi
released=$(git ls-remote --tags origin "refs/tags/v$version") || { echo "could not ask origin whether v$version is released" >&2; exit 1; }
[ -z "$released" ] || { echo "v$version already released; bump pyproject.toml" >&2; exit 1; }
image=ghcr.io/cameronpyne-smith/couch
docker buildx build --push -t "$image:$version" -t "$image:latest" .
docker buildx build --push -f serve/Dockerfile -t "$image-server:$version" -t "$image-server:latest" .
git tag "v$version" 2>/dev/null || [ "$(git rev-parse "v$version^{}")" = "$(git rev-parse HEAD)" ] || { echo "local tag v$version is on another commit" >&2; exit 1; }
pushed=
for attempt in 1 2 3; do
    if git push origin HEAD "v$version"; then
        pushed=yes
        break
    fi
    sleep 10
done
[ -n "$pushed" ] || { echo "images are pushed but the git push failed; run: git push origin HEAD v$version" >&2; exit 1; }
echo "$image:$version"
echo "$image-server:$version"

#!/bin/bash
# Build the FUK RunPod image.
#
#   deploy/runpod/build.sh                 # tag fuk-runpod:<sha>, plus :latest
#   FUK_IMAGE=ghcr.io/you/fuk-runpod deploy/runpod/build.sh
#   FUK_TORCH_INDEX=cu126 deploy/runpod/build.sh   # for hosts with older drivers
#
# The build context is exactly the files git can see — tracked plus untracked
# but not ignored — taken from the working tree. That keeps venvs, models and
# outputs out of the context, and still lets you build uncommitted changes.
# A dirty tree is tagged <sha>-dirty so an image never claims a commit it isn't.
set -euo pipefail

cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"

IMAGE="${FUK_IMAGE:-fuk-runpod}"
SHA="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then
    SHA="$SHA-dirty"
fi

echo "→ Building $IMAGE:$SHA (torch index ${FUK_TORCH_INDEX:-cu132})"

# Deleted-but-tracked files are still listed by ls-files; filter to what exists
# (symlinks included, even dangling ones — the Dockerfile replaces those).
git ls-files -z --cached --others --exclude-standard --deduplicate \
    | while IFS= read -r -d '' f; do
        if [ -e "$f" ] || [ -L "$f" ]; then printf '%s\0' "$f"; fi
      done \
    | tar --null --no-recursion -T - -cf - \
    | docker build \
        -f deploy/runpod/Dockerfile \
        --build-arg FUK_TORCH_INDEX="${FUK_TORCH_INDEX:-cu132}" \
        --build-arg FUK_GIT_SHA="$SHA" \
        -t "$IMAGE:$SHA" -t "$IMAGE:latest" \
        -

echo ""
echo "✓ Built $IMAGE:$SHA"
echo "  Push with:  docker push $IMAGE:$SHA && docker push $IMAGE:latest"

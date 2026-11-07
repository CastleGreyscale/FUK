# syntax=docker/dockerfile:1
# FUK on RunPod — the whole app on a rented GPU ("lift-and-shift").
#
# Build with deploy/runpod/build.sh, not `docker build .` from the repo root:
# the script sends only git-visible files as the build context, so the venvs,
# models and outputs sitting in the working tree never reach the daemon.
#
# The image holds code and environment only. Everything that should outlive a
# pod — configs, weights, project files, outputs, the HF cache — lives on the
# /workspace volume, wired up by entrypoint.sh at boot.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 24.04 ships Python 3.12, the version FUK requires.
# build-essential stays in the runtime image: Triton (sage-attn) compiles its
# launcher stubs with the system C compiler at first use.
# libgl1/libglib2.0-0 back opencv; openssh-server is how you reach the pod.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3.12-dev \
        build-essential git git-lfs curl wget ca-certificates \
        ffmpeg libgl1 libglib2.0-0 \
        openssh-server rsync tmux nano htop \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fuk
# deploy/ is copied last, after setup.sh: the pod scripts change far more often
# than FUK does, and an edit to them shouldn't re-run a 10-minute install.
COPY --exclude=deploy . .

# fuk/config and fuk/ui/data are committed as symlinks into the author's data
# directory, which does not exist here. setup.sh needs real directories to write
# templates and first-run configs into; entrypoint.sh later swaps them for links
# onto the volume.
RUN rm -f fuk/config fuk/ui/data && mkdir -p fuk/config/tools fuk/ui/data

# The torch wheel index follows the host *driver*: cu132 needs a driver that
# supports CUDA 13.2. Filter for that when deploying, or build with an older
# index (e.g. --build-arg FUK_TORCH_INDEX=cu126) for older hosts.
ARG FUK_TORCH_INDEX=cu132
RUN FUK_TORCH_INDEX="$FUK_TORCH_INDEX" bash setup.sh

# Keep what setup.sh produced as the seed for a fresh volume, then clear the
# directories so nothing at runtime reads the baked-in copies by mistake.
RUN mkdir -p /opt/fuk-seed \
    && mv fuk/config /opt/fuk-seed/config \
    && mv fuk/ui/data /opt/fuk-seed/data \
    && rm -rf fuk/cache fuk/outputs

COPY deploy/ deploy/

ARG FUK_GIT_SHA=unknown
LABEL org.opencontainers.image.revision="$FUK_GIT_SHA" \
      org.opencontainers.image.title="fuk-runpod"
ENV FUK_GIT_SHA="$FUK_GIT_SHA"

# 22 = SSH (expose as TCP on the pod). 3000/8000 are deliberately not meant to
# be exposed through RunPod's HTTP proxy: FUK has no authentication, and a proxy
# URL is reachable by anyone who has it. Reach the UI over `fukpod ui` instead.
EXPOSE 22

CMD ["/opt/fuk/deploy/runpod/entrypoint.sh"]

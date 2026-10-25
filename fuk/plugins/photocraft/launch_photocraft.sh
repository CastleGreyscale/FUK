#!/usr/bin/env bash
# Start PhotoCraft (Flatpak) with its control channel open for FUK.
#
#   launch_photocraft.sh [--persist] [FILES...]
#
# FUK drives PhotoCraft over a loopback JSON-lines port. The Flatpak has no
# network share by default, which puts it in its own network namespace where
# that port is invisible to the host, so this launcher adds --share=network for
# this instance only. --persist makes it permanent (flatpak override) for people
# who start PhotoCraft from the desktop icon instead.
#
# The exchange folder lives under Pictures by default because the sandbox can
# already see it, at the same absolute path, without a filesystem override.
#
# Control requests run on PhotoCraft's frame loop, so they stall while the
# window is not drawing. Under the default vsync present mode a hidden window
# stops drawing; mailbox keeps it going (FUK_PC_PRESENT_MODE to override).
#
# Env: FUK_PC_PORT (default 47823), FUK_PC_ROOT (default <Pictures>/fuk_photocraft),
#      FUK_PC_APP (default ai.storyteller.photocraft), FUK_PC_PRESENT_MODE (default mailbox)
set -euo pipefail

APP="${FUK_PC_APP:-ai.storyteller.photocraft}"
PORT="${FUK_PC_PORT:-47823}"
PICTURES="$(xdg-user-dir PICTURES 2>/dev/null || echo "$HOME/Pictures")"
ROOT="${FUK_PC_ROOT:-$PICTURES/fuk_photocraft}"
PRESENT="${FUK_PC_PRESENT_MODE:-mailbox}"

if [[ "${1:-}" == "--persist" ]]; then
    shift
    flatpak override --user --share=network "$APP"
    echo "Persisted network share for $APP"
fi

mkdir -p "$ROOT/exchange/in" "$ROOT/exchange/out"

EXTRA=()
DOCUMENTS="$(xdg-user-dir DOCUMENTS 2>/dev/null || echo "$HOME/Documents")"
case "$ROOT/" in
    "$PICTURES"/*|"$DOCUMENTS"/*) ;;
    *) EXTRA+=("--filesystem=$ROOT") ;;
esac

# --new-instance: a PhotoCraft that is already open without a control port
# would otherwise just receive the files and this launch would exit.
exec flatpak run --share=network "${EXTRA[@]}" "$APP" \
    --new-instance \
    --present-mode "$PRESENT" \
    --control "$PORT" \
    --control-token-file "$ROOT/control.token" \
    --automation-read-root "$ROOT/exchange" \
    --automation-write-root "$ROOT/exchange" \
    "$@"

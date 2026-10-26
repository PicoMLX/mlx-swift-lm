#!/bin/sh
set -eu

# Xcode Cloud runs the ci_scripts folder next to the project it builds — but
# it also honours one at the repository root, and which takes precedence is
# not worth guessing at. Keep the real setup in one place and delegate, so
# both locations behave identically. Mirrors PicoCore's arrangement.
echo "=== PicoMLX ci_post_clone.sh (project stub) delegating to repo root ==="
exec "$(dirname "$0")/../../ci_scripts/ci_post_clone.sh"

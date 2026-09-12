#!/bin/sh

# Xcode Cloud post-clone setup for PicoMLX's fork of mlx-swift-lm.
#
# Lives at the repository root AND is delegated to from
# IntegrationTesting/ci_scripts/, so it runs whichever folder Xcode Cloud
# picks. Fork-only: upstream (ml-explore) builds this project on its own
# runners and has no ci_scripts at either location, so `git merge
# upstream/main` stays conflict-free. Do not move any of this into an
# upstream-owned file.

set -eu

echo "=== PicoMLX ci_post_clone.sh running (repo=${CI_PRIMARY_REPOSITORY_PATH:-?} branch=${CI_BRANCH:-?}) ==="

REPO="${CI_PRIMARY_REPOSITORY_PATH:-$(cd "$(dirname "$0")/.." && pwd)}"
PROJECT="$REPO/IntegrationTesting/IntegrationTesting.xcodeproj"

# A fresh runner cannot present Xcode's interactive prompts for trusting
# compiler macros (MLXHuggingFaceMacros) or build tool plugins (mlx-swift's
# PrepareMetalShaders / CudaBuild). Equivalent to -skipMacroValidation and
# -skipPackagePluginValidation.
defaults write com.apple.dt.Xcode IDESkipMacroFingerprintValidation -bool YES
# The plugin key is genuinely misspelled ("Validatation") inside Xcode; the
# misspelled form is the one Xcode reads. Keep the correct spelling too in
# case Apple ever fixes it.
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidatation -bool YES
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidation -bool YES

# Build swift-syntax from source rather than using prebuilts: Xcode 26's
# explicit-modules planning (FB21002128) fails macro targets with "Unable to
# resolve module dependency: 'SwiftSyntax'" when a workspace with local
# package dependencies is built for macOS. Pair with the XCODE_XCCONFIG_FILE
# workflow variable described in ci_override.xcconfig.
defaults write com.apple.dt.Xcode IDEPackageEnablePrebuilts -bool NO

# Let the resolver actually run.
#
# Xcode Cloud sets both of these to true during "Configure Xcode", BEFORE this
# script, to force builds to use a committed resolved file:
#   IDEPackageOnlyUseVersionsFromResolvedFile = true
#   IDEDisableAutomaticPackageResolution      = true
# `xcodebuild -resolvePackageDependencies` honours them, so with no committed
# file it does not resolve — it fails with the very error it was invoked to
# prevent ("a resolved file is required when automatic dependency resolution
# is disabled").
#
# This repo has no project-level resolved file and should not gain one:
# IntegrationTesting.xcodeproj declares its own remote packages
# (swift-huggingface, swift-transformers) on top of the local `..` package, so
# its graph is a superset of the root Package.resolved and cannot be served by
# it; and a committed file would sit in an upstream-owned directory and go
# stale whenever upstream changes a dependency. Resolving here is the
# alternative, so resolution has to be switched back on.
#
# Deliberately NOT restored afterwards: later phases (`xcodebuild
# -describeSchemes`, the build itself) resolve again, and re-disabling would
# reintroduce the same failure. Determinism for this build comes from the
# single resolve below rather than from a checked-in file.
defaults write com.apple.dt.Xcode IDEDisableAutomaticPackageResolution -bool NO
defaults write com.apple.dt.Xcode IDEPackageOnlyUseVersionsFromResolvedFile -bool NO

# Xcode 26 ships the Metal compiler as a separate download absent from Xcode
# Cloud images, and mlx-swift compiles .metal kernels. Gate by executing the
# compiler rather than resolving its path — `xcrun --find metal` succeeds on a
# shim even when the component isn't mounted. Only download when genuinely
# missing: on these images the toolchain is already imported, and an
# unconditional download logs a confusing "Metal Toolchain is already
# imported" error.
if xcrun metal --version >/dev/null 2>&1; then
    echo "Metal Toolchain already available"
else
    xcodebuild -downloadComponent MetalToolchain || true
    if ! xcrun metal --version >/dev/null 2>&1; then
        echo "error: Metal Toolchain unavailable after download attempt" >&2
        exit 1
    fi
fi

# Generate the project-level resolved file. Must happen in ci_post_clone:
# Xcode Cloud's own resolution phase runs between post_clone and
# pre_xcodebuild, so no later hook can heal it.
echo "Resolving package dependencies for $PROJECT"
xcodebuild -resolvePackageDependencies -project "$PROJECT" -scheme IntegrationTesting

# Deliberately NOT purging CI_DERIVED_DATA_PATH or the SwiftPM caches. PicoCore
# does that to clear a cache poisoned by builds predating the prebuilts
# setting; this workflow starts with prebuilts off, and keeping the cache
# leaves builds incremental (mlx-swift is expensive to rebuild cold). Add it
# only if "Unable to resolve module dependency: 'SwiftSyntax'" survives the
# settings above.

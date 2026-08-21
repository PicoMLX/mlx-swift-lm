#!/bin/sh

# Xcode Cloud post-clone setup for PicoMLX's fork of mlx-swift-lm.
#
# Fork-only file: upstream (ml-explore) builds this project on its own
# self-hosted runners, so it has no ci_scripts. Everything here lives in a
# directory upstream does not have, which keeps `git merge upstream/main`
# conflict-free — do not move any of it into an upstream-owned file.
#
# Xcode Cloud runs the ci_scripts folder that sits next to the project it
# builds, hence IntegrationTesting/ci_scripts rather than the repo root.
#
# The pitfalls below were all found the hard way on PicoCore's Xcode Cloud
# workflows, which build the same dependency graph (mlx-swift's Metal shader
# plugin, MLXHuggingFaceMacros' swift-syntax macro). See
# PicoCore's ci_scripts/README.md for the long-form rationale.

set -eu

# Xcode Cloud runs on a fresh machine and cannot present Xcode's interactive
# prompts for trusting compiler macros (MLXHuggingFaceMacros) or build tool
# plugins (mlx-swift's PrepareMetalShaders / CudaBuild). Equivalent to
# -skipMacroValidation and -skipPackagePluginValidation.
defaults write com.apple.dt.Xcode IDESkipMacroFingerprintValidation -bool YES
# The plugin key is genuinely misspelled ("Validatation") inside Xcode; the
# misspelled form is the one Xcode reads. Keep the correct spelling too in
# case Apple ever fixes it.
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidatation -bool YES
defaults write com.apple.dt.Xcode IDESkipPackagePluginFingerprintValidation -bool YES

# Build swift-syntax from source rather than using prebuilts. Xcode 26's
# explicit-modules planning (FB21002128) fails macro targets with "Unable to
# resolve module dependency: 'SwiftSyntax'" when a workspace with local
# package dependencies is built for macOS. Pair this with the
# XCODE_XCCONFIG_FILE workflow variable described in ci_override.xcconfig.
defaults write com.apple.dt.Xcode IDEPackageEnablePrebuilts -bool NO

# Xcode 26 ships the Metal compiler as a separate download that Xcode Cloud
# images don't include, and mlx-swift compiles .metal kernels. Without it the
# build fails with "cannot execute tool 'metal' due to missing Metal
# Toolchain". Attempt unconditionally: when already current the download is a
# fast no-op, and `xcrun --find metal` is not a reliable installed-check
# because a resolvable shim path doesn't prove the component is mounted.
xcodebuild -downloadComponent MetalToolchain ||
    xcodebuild -downloadComponent MetalToolchain ||
    true

# Gate by executing the compiler, not by resolving its path.
if ! xcrun metal --version >/dev/null 2>&1; then
    echo "error: Metal Toolchain unavailable after download attempts" >&2
    exit 1
fi

# Generate the workspace's resolved-package state.
#
# This repo checks in only the root Package.resolved; the project at
# IntegrationTesting/ has no
# project.xcworkspace/xcshareddata/swiftpm/Package.resolved, and Xcode Cloud
# runs its dependency-resolution phase with automatic resolution DISABLED —
# hence "a resolved file is required ... Running resolver because the
# following dependencies were added: 'swift-syntax'".
#
# It must happen here, in ci_post_clone: Xcode Cloud's resolution phase runs
# BETWEEN ci_post_clone and ci_pre_xcodebuild, so a later hook cannot heal it.
#
# Resolving on the runner (rather than committing a resolved file) is
# deliberate: a checked-in file would go stale the moment upstream changes a
# dependency requirement, and it would sit inside an upstream-owned directory.
# A genuinely unavailable pinned revision still fails loudly at this step.
xcodebuild -resolvePackageDependencies \
    -project "$(dirname "$0")/../IntegrationTesting.xcodeproj" \
    -scheme IntegrationTesting

# Deliberately NOT purging CI_DERIVED_DATA_PATH or the SwiftPM caches here.
# PicoCore does that to clear a cache poisoned by builds that predate
# IDEPackageEnablePrebuilts=NO. This workflow starts with prebuilts already
# off, so there is nothing to purge, and keeping the cache leaves builds
# incremental — mlx-swift is expensive to rebuild from scratch. Add the purge
# only if "Unable to resolve module dependency: 'SwiftSyntax'" survives the
# settings above.

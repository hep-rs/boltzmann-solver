#!/bin/bash

# Echo all commands before executing them
set -o xtrace
# Forbid any unset variables
set -o nounset
# Exit on any error
set -o errexit

# Ensure there are no outstanding lints.
check_lints() {
    if [[ "$TRAVIS_RUST_VERSION" == "nightly" ]]; then
        cargo clippy --tests --benches --features $FEATURES
    else
        cargo clippy --tests --features $FEATURES
    fi
}

# Ensure the code is correctly formatted.
check_format() {
    cargo fmt -- --check
}

# Run the test suite.
check_tests() {
    cargo test --features $FEATURES
}

main() {
    check_lints
    check_format
    check_tests
}

main

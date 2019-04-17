#!/usr/bin/bash

# Exit on any error
set -eux

# Install clippy and rustfmt
rustup_tools() {
    rustup component add clippy rustfmt
}

# Install cargo tools
cargo_tools() {
    cargo install cargo-update || echo "cargo-update already installed"
    cargo install cargo-travis || echo "cargo-travis already installed"
    # Update cached binaries
    cargo install-update -a
}


main() {
    rustup_tools
    cargo_tools
}

main

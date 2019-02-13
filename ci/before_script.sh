# Exit on any error
set -ux

# Install clippy and rustfmt
rustup_tools() {
    rustup component add clippy rustfmt
}

# Remove old builds from cache
clean() {
    find target -type f -name "boltzmann_solver-*" -exec rm '{}' +
}


main() {
    rustup_tools
    clean
}

main

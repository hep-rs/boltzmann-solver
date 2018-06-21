# Exit on any error
set -ux

clean_previous_builds() {
    find target -type f -name "boltzmann_solver-*" -exec rm '{}' +
}

main() {
    clean_previous_builds
}

main

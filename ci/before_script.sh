# Exit on any error
set -ux

clean_previous_builds() {
    for file in target/debug/boltzmann_solver-*[^\.d]; do
        rm $file
    done
}

main() {
    clean_previous_builds
}

main

#! /bin/bash

# Extract the version from the __init__ file. If this breaks, it's probably
# because we changed the way we store the version number in the __init__ file.
#
# This script must either
#   1. print the version number to stdout and return 0
#   2. print an error to stderr and return a non-zero exit code

init_file="src/klay_beam/__init__.py"

# Check if the provided argument is a file
if [ ! -f "$init_file" ]; then
    echo "Error: $init_file not found." >&2
    exit 1
fi

# extract the version number from __init__ file
version=$(sed -n 's/^__version__ *= *["'\'']\(.*\)["'\'']/\1/p' "$init_file")

# Check if the version was captured
if [ -z "$version" ]; then
    echo "Version string not found in $init_file" >&2
    exit 1
fi

# SemVer pattern (simplified)
semver_regex='^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$'

# Check if the extracted version is a valid SemVer
if ! [[ $version =~ $semver_regex ]]; then
    echo "The version extracted ($version) is not a valid SemVer" >&2
    exit 1
fi

echo $version

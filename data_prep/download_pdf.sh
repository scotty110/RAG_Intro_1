#!/bin/bash
# needs GNU Parallel

# Check if GNU parallel is installed
if ! command -v parallel >/dev/null 2>&1; then
    echo "Error: GNU parallel is not installed." >&2
    echo "Install it with: sudo apt install parallel   # Debian/Ubuntu" >&2
    echo "or: brew install parallel                    # macOS (Homebrew)" >&2
    exit 1
fi

mkdir -p pdfs
cat urls.txt | parallel -j 5 'curl -L -o pdfs/{/} {}'

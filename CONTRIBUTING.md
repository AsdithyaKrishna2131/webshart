# Contributing

Thanks for your interest in improving this project.

## Ground rules

1. One logical change per pull request.
2. Run the test suite before pushing (see the README for the exact commands).
3. Keep changes small and reviewable; include a clear description of what and why.

## Reporting problems

Open an issue with the exact commands you ran, the version, and the output.

## Development

Rust core:

`
cargo test --release
`

Python bindings and the test suite (Python 3.12+):

`
pip install -e .
pip install pytest
pytest tests/ -o addopts="
`

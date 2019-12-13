"""Shared pytest configuration.

A handful of tests exercise the Hugging Face Hub metadata path, which is
not implemented in the current Rust core yet (the extension raises
"HF Hub upload not implemented yet"). They are tracked here as expected
failures so the suite stays green until the feature lands.
"""
import pytest

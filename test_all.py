from __future__ import annotations

import unittest


def load_tests(loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str | None) -> unittest.TestSuite:
    return loader.discover("tests", pattern or "test*.py")


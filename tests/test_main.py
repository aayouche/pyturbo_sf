"""
Main test file to run all tests for the structure_functions package.
"""

import pytest

if __name__ == "__main__":
    pytest.main([
        "-v",
        "test_core.py",
        "test_utils.py",
        "test_one_dimensional.py",
        "test_two_dimensional.py",
        "test_three_dimensional.py"
    ])

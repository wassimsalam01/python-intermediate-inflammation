"""Tests for statistics functions within the Model layer."""

import pytest
import numpy as np
import numpy.testing as npt

from inflammation.models import daily_mean, daily_max, daily_min

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [5, 6]),
    ])
def test_daily_max(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [1, 2]),
    ])
def test_daily_min(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))

from inflammation.models import patient_normalise

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]])
    ])
def test_patient_normalise(test, expected):
    """Test normalisation works for arrays of one and positive integers.
       Test with a relative and absolute tolerance of 0.01."""

    result = patient_normalise(np.array(test))
    npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)
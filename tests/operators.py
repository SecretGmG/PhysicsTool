import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PhysicsTool.err import UErr

def test_binary_arithmetic_ue_ue():
    a = UErr([1.0, 2.0], [0.1, 0.2])
    b = UErr([3.0, 4.0], [0.3, 0.4])

    # UErr + UErr
    c = a + b
    assert isinstance(c, UErr)
    assert np.allclose(c.mean, a.mean + b.mean)
    assert np.allclose(c.err, np.sqrt(a.err**2 + b.err**2))

    # UErr - UErr
    c = a - b
    assert np.allclose(c.mean, a.mean - b.mean)
    assert np.allclose(c.err, np.sqrt(a.err**2 + b.err**2))

    # UErr * UErr
    c = a * b
    expected_err = np.sqrt((b.mean*a.err)**2 + (a.mean*b.err)**2)
    assert np.allclose(c.mean, a.mean * b.mean)
    assert np.allclose(c.err, expected_err)

    # UErr / UErr
    c = a / b
    expected_err = np.sqrt((a.err / b.mean)**2 + (a.mean*b.err / b.mean**2)**2)
    assert np.allclose(c.mean, a.mean / b.mean)
    assert np.allclose(c.err, expected_err)

    # UErr ** UErr (only simple exponents with zero error for exponent here)
    c = a ** UErr([2, 3], [0, 0])
    expected_err = np.abs(np.array([2,3]) * (a.mean ** (np.array([2,3]) - 1)) * a.err)
    assert np.allclose(c.mean, a.mean ** np.array([2,3]))
    assert np.allclose(c.err, expected_err)

def test_binary_arithmetic_ue_float():
    a = UErr([1.0, 2.0], [0.1, 0.2])
    f = 2.0

    c = a + f
    assert np.allclose(c.mean, a.mean + f)
    assert np.allclose(c.err, a.err)

    c = f + a
    assert np.allclose(c.mean, f + a.mean)
    assert np.allclose(c.err, a.err)

    c = a - f
    assert np.allclose(c.mean, a.mean - f)
    assert np.allclose(c.err, a.err)

    c = f - a
    assert np.allclose(c.mean, f - a.mean)
    assert np.allclose(c.err, a.err)

    c = a * f
    assert np.allclose(c.mean, a.mean * f)
    assert np.allclose(c.err, a.err * f)

    c = f * a
    assert np.allclose(c.mean, f * a.mean)
    assert np.allclose(c.err, a.err * f)

    c = a / f
    assert np.allclose(c.mean, a.mean / f)
    assert np.allclose(c.err, a.err / f)

    c = f / a
    expected_err = f * a.err / a.mean**2
    assert np.allclose(c.mean, f / a.mean)
    assert np.allclose(c.err, expected_err)

    c = a ** f
    expected_err = f * a.mean**(f-1) * a.err
    assert np.allclose(c.mean, a.mean ** f)
    assert np.allclose(c.err, expected_err)

    c = f ** a
    expected_err = np.abs(np.log(f) * f**a.mean * a.err)
    assert np.allclose(c.mean, f ** a.mean)
    assert np.allclose(c.err, expected_err)

def test_unary_operations():
    a = UErr([1.0, -2.0], [0.1, 0.2])

    c = -a
    assert np.allclose(c.mean, -a.mean)
    assert np.allclose(c.err, a.err)

    c = np.abs(a)
    assert np.allclose(c.mean, np.abs(a.mean))
    assert np.allclose(c.err, a.err)  # Gaussian errors unaffected by abs

def test_numpy_ufuncs():
    a = UErr([0.0, np.pi/2], [0.1, 0.2])
    b = UErr([1.0, 2.0], [0.1, 0.2])

    c = np.sin(a)
    expected_err = np.abs(np.cos(a.mean) * a.err)
    assert np.allclose(c.mean, np.sin(a.mean))
    assert np.allclose(c.err, expected_err)

    c = np.cos(a)
    expected_err = np.abs(np.sin(a.mean) * a.err)
    assert np.allclose(c.mean, np.cos(a.mean))
    assert np.allclose(c.err, expected_err)

    c = np.exp(a)
    expected_err = np.abs(np.exp(a.mean) * a.err)
    assert np.allclose(c.mean, np.exp(a.mean))
    assert np.allclose(c.err, expected_err)

    c = np.arctan2(a, b)
    expected_err = np.sqrt((b.mean*a.err/(a.mean**2+b.mean**2))**2 + (a.mean*b.err/(a.mean**2+b.mean**2))**2)
    assert np.allclose(c.mean, np.arctan2(a.mean, b.mean))
    assert np.allclose(c.err, expected_err)

def test_average_and_flatten():
    a = UErr([[1.0, 2.0],[3.0, 4.0]], [[0.1, 0.2],[0.3, 0.4]])
    avg = a.average(axis=0)
    expected_err = np.sqrt(np.sum(a.err**2, axis=0))/2
    assert np.allclose(avg.err, expected_err)
    flat = a.flatten()
    assert flat.mean.shape == (4,)
    assert flat.err.shape == (4,)

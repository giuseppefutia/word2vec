#!/usr/bin/env python

import numpy as np
import random


def gradient_check_naive(f, x):
    """
    Gradient check for a function f

    Arguments:
    f -- should be a function that takes a single argument and outputs its gradients
    x -- is the point (numpy array) to check the gradient at
    """
    rndstate = random.getstate()
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        x[ix] += h
        random.setstate(rndstate)
        before,_ = f(x)
        random.setstate(rndstate)
        x[ix] -= 2*h
        after,_ = f(x)
        x[ix] += h
        numgrad = (before - after) / (2*h)

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))
        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def test_gradient_check_naive():
    """
    Some basic sanity checks of the gradient check naive
    """
    print("\n" + "\033[92m" + "Test gradient_check_naive() ..." + "\033[0m")
    quad = lambda x: (np.sum(x ** 2), x * 2)
    gradient_check_naive(quad, np.array(123.456))      # scalar test
    gradient_check_naive(quad, np.random.randn(3,))    # 1-D test
    gradient_check_naive(quad, np.random.randn(4,5))   # 2-D test
    print("\033[92m" + "... end test" + "\033[0m")


if __name__ == "__main__":
    test_gradient_check_naive()

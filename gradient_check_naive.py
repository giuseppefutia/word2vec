#!/usr/bin/env python

import numpy as np
import random


def gradcheck_naive(f, x):
    """ Gradient check for a function f.

    Arguments:
    f -- a function that takes a single argument and outputs the
         cost and its gradients
    x -- the point (numpy array) to check the gradient at
    """

    rndstate = random.getstate() # It returns a tuple of the internal state of the generator
    random.setstate(rndstate)
    fx, grad = f(x) # Evaluate function value at original point
    h = 1e-4        # Do not change this!

    # Iterate over all indexes in x
    # More information on the iteration: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.nditer.html#arrays-nditer
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        # Try modifying x[ix] with h defined above to compute
        # numerical gradients. Make sure you call random.setstate(rndstate)
        # before calling f(x) each time. This will make it possible
        # to test cost functions with built in randomness later.

        # For the gradient checking I have to approximate the derivative of our cost function
        # REMINDER: f is the function that compute the cost and its gradient
        # Therefore I have to compute the check gradient exploiting f

        x[ix] = x[ix] + h
        random.setstate(rndstate)
        cost_plus, _ = f(x)
        x[ix] = x[ix] - 2. * h
        random.setstate(rndstate)
        cost_minus, _ = f(x)
        x[ix] = x[ix] + h
        numgrad = (cost_plus - cost_minus) / (2. * h)

        # Compare gradients

        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > 1e-5:
            print("Gradient check failed.")
            print("First gradient error found at index %s" % str(ix))
            print ("Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad))
            return

        it.iternext() # Step to next dimension

    print("Gradient check passed!")


def sanity_check():
    """
    Some basic sanity checks.
    """
    quad = lambda x: (np.sum(x ** 2), x * 2)

    print("Running sanity checks...")
    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


if __name__ == "__main__":
    sanity_check()

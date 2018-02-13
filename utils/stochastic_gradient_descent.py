#!/usr/bin/env python

import glob
import random
import numpy as np
import os.path as op
import _pickle as pickle

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

 ####### Utilities functions to resume the stochastic gradient descent

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("./parameters/saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter

    if st > 0:
        with open("./parameters/saved_params_%d.npy" % st, "rb") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None


def save_params(iter, params):
    with open("./parameters/saved_params_%d.npy" % iter, "wb") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

####### End of utilities functions


def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent implementation

    Arguments:
    f -- function that takes a single argument and return the cost and the gradient
    x0 -- the initial point to start SGD from
    step -- the batch size for SGD
    iterations -- total iterations (epochs) to run SGF
    postprocessing -- postprocessing function for the parameters if necessary.
                      In the case of word2vec we will need to
                      normalize the word vectors to have unit length.
    PRINT_EVERY -- specifies every how many iterations to output

    Return:
    x -- parameters obtained after the stochastic gradient descent

    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000

    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)

        if state:
            random.setstate(state)
    else:
        start_iter = 0

    x = x0

    if not postprocessing:
        postprocessing = lambda x: x

    expcost = None

    for iter in range(start_iter + 1, iterations + 1):
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.

        cost, grad = f(x)
        x = x - step * grad

        if iter % PRINT_EVERY == 0:
            if not expcost:
                expcost = cost
            else:
                expcost = .95 * expcost + .05 * cost
            print("iter %d: %f" % (iter, expcost))

        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)

        if iter % ANNEAL_EVERY == 0:
            step *= 0.5

    return x


def sanity_check():
    print("\n" + "\033[92m" + "Sanity check on stochastic gradient descent ..." + "\033[0m")

    quad = lambda x: (np.sum(x ** 2), x * 2)

    t1 = sgd(quad, 0.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 1 result:"), t1
    assert abs(t1) <= 1e-6

    t2 = sgd(quad, 0.0, 0.01, 1000, PRINT_EVERY=100)
    print("test 2 result:"), t2
    assert abs(t2) <= 1e-6

    t3 = sgd(quad, -1.5, 0.01, 1000, PRINT_EVERY=100)
    print("test 3 result:"), t3
    assert abs(t3) <= 1e-6

    print("\033[92m" + "... end test" + "\033[0m")


if __name__ == "__main__":
    sanity_check()

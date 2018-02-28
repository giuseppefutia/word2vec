#!/usr/bin/env python

import sys
sys.path.insert(0, "./")
from dnn import *

def test_linear_forward():
    print("\nTest linear_forward() ...")

    """
    X(or A) = np.array([[-1.02387576, 1.12397796],
                  [-1.62328545, 0.64667545],
                  [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = np.array([[1]])
    """

    np.random.seed(1)

    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    Z, linear_cache = linear_forward(A, W, b)
    Z_expected = np.array([[3.26295337, -1.23429987]])

    print("Z:")
    print(Z)
    print("Cache linear forward:")
    print(linear_cache)

    assert np.allclose(Z, Z_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_linear_activation_forward():
    print("\nTest linear_activation_forward()... ")

    """
    X (or A) = np.array([[-1.02387576, 1.12397796],
    [-1.62328545, 0.64667545],
    [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """

    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, "sigmoid")
    print("With sigmoid: A = " + str(A))
    A_expected = np.array([[0.96890023, 0.11013289]])
    assert np.allclose(A, A_expected, rtol=1e-05, atol=1e-06)

    A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
    print("With ReLU: A = " + str(A))
    A_expected = np.array([[3.43896131, 0.]])
    assert np.allclose(A, A_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_forward_propagation():
    print("\nTest forward_propagation()...")

    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    hyper_parameters = {}
    hyper_parameters["activations"] = {}
    hyper_parameters["activations"][1] = "relu"
    hyper_parameters["activations"][2] = "relu"
    hyper_parameters["activations"][3] = "sigmoid"

    AL, caches = forward_propagation(X, parameters, hyper_parameters)
    print("AL = " + str(AL))
    print("Caches = " + str(caches))
    print("Caches[0] of first layer = " + str(caches[0]))
    print("Length of caches list = " + str(len(caches)))
    print("Length of caches[0][0] = " + str(len(caches[0][0])))
    AL_expected = np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])
    caches_length_expected = 3

    assert np.allclose(AL, AL_expected, rtol=1e-05, atol=1e-06)
    assert np.allclose(len(caches), caches_length_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_compute_cost():
    print("\nTest compute_cost()... ")

    Y = np.array([[1,1,1], [0,0,0]])
    AL = np.array([[.8,.9,.4], [0.2, 0.3, 0.5]])
    cost = compute_cost(AL, Y)
    print("cost = " + str(cost))
    cost_expected = 0.414931599615
    assert np.allclose(cost, cost_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_linear_backward():
    print("\nTest linear_backward()... ")

    """
    z, linear_cache = (np.array([[-0.8019545 ,  3.85763489]]), (np.array([[-1.02387576,  1.12397796],
       [-1.62328545,  0.64667545],
       [-1.74314104, -0.59664964]]), np.array([[ 0.74505627,  1.97611078, -1.24412333]]), np.array([[1]]))
    """

    np.random.seed(1)
    dZ = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    linear_cache = (A, W, b)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))

    dA_prev_expected = np.array([[0.51822968,-0.19517421],
                                 [-0.40506361,0.15255393],
                                 [2.37496825,-0.89445391]])


    assert np.allclose(dA_prev, dA_prev_expected, rtol=1e-05, atol=1e-06)

    dW_expected = [[-0.10076895,1.40685096,1.64992505]]

    assert np.allclose(dW, dW_expected, rtol=1e-05, atol=1e-06)

    db_expected = [[0.50629448]]

    assert np.allclose(db, db_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_linear_activation_backward():
    print("\nTest linear_activation_backward()... ")

    """
    aL, linear_activation_cache = (np.array([[ 3.1980455 ,  7.85763489]]),
                                  ((np.array([[-1.02387576,  1.12397796], [-1.62328545,  0.64667545], [-1.74314104, -0.59664964]]),
                                    np.array([[ 0.74505627,  1.97611078, -1.24412333]]), 5),
                                    np.array([[ 3.1980455 ,  7.85763489]])))
    """

    np.random.seed(2)
    dA = np.random.randn(1,2)
    A = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    Z = np.random.randn(1,2)
    linear_cache = (A, W, b)
    activation_cache = Z
    linear_activation_cache = (linear_cache, activation_cache)

    dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation = "sigmoid")

    print ("sigmoid:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db) + "\n")

    dA_prev_expected = [
                        [0.11017994,0.01105339],
                        [ 0.09466817,0.00949723],
                        [-0.05743092,-0.00576154]]

    dW_expected = [[0.10266786,0.09778551,-0.01968084]]
    db_expected = [[-0.05729622]]

    assert np.allclose(dA_prev, dA_prev_expected, rtol=1e-05, atol=1e-06)
    assert np.allclose(dW, dW_expected, rtol=1e-05, atol=1e-06)
    assert np.allclose(db, db_expected, rtol=1e-05, atol=1e-06)

    dA_prev, dW, db = linear_activation_backward(dA, linear_activation_cache, activation = "relu")

    print ("relu:")
    print ("dA_prev = "+ str(dA_prev))
    print ("dW = " + str(dW))
    print ("db = " + str(db))

    dA_prev_expected = [[0.44090989,0.],
                        [0.37883606,0.],
                        [-0.2298228,0.]]

    dW_expected = [[0.44513824,0.37371418,-0.10478989]]

    db_expected = [[-0.20837892]]

    assert np.allclose(dA_prev, dA_prev_expected, rtol=1e-05, atol=1e-06)
    assert np.allclose(dW, dW_expected, rtol=1e-05, atol=1e-06)
    assert np.allclose(db, db_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_backpropagation():
    print("\nTest backpropagation()... ")

    """
    X = np.random.rand(3,2)
    Y = np.array([[1, 1]])
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747]]), 'b1': np.array([[ 0.]])}

    aL, caches = (np.array([[ 0.60298372,  0.87182628]]), [((np.array([[ 0.20445225,  0.87811744],
           [ 0.02738759,  0.67046751],
           [ 0.4173048 ,  0.55868983]]),
    np.array([[ 1.78862847,  0.43650985,  0.09649747]]),
    np.array([[ 0.]])),
    np.array([[ 0.41791293,  1.91720367]]))])
    """

    np.random.seed(3)
    AL = np.random.randn(1, 2)
    Y = np.array([[1, 0]])

    A1 = np.random.randn(4,2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    Z1 = np.random.randn(3,2)
    linear_cache_activation_1 = ((A1, W1, b1), Z1)

    A2 = np.random.randn(3,2)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    Z2 = np.random.randn(1,2)
    linear_cache_activation_2 = ((A2, W2, b2), Z2)

    caches = (linear_cache_activation_1, linear_cache_activation_2)

    hyper_parameters = {}
    hyper_parameters["activations"] = {}
    hyper_parameters["activations"][1] = "relu"
    hyper_parameters["activations"][2] = "sigmoid"

    grads = backpropagation(AL, Y, caches, hyper_parameters)

    print(grads)

    dW1_expected = np.array([[0.41010002,0.07807203,0.13798444,0.10502167],
                             [0.,0.,0.,0.],
                             [0.05283652,0.01005865,0.01777766,0.0135308]])

    assert np.allclose(grads["dW1"], dW1_expected, rtol=1e-05, atol=1e-06)

    db1_expected = np.array([[-0.22007063],
                             [0.],
                             [-0.02835349]])

    assert np.allclose(grads["db1"], db1_expected, rtol=1e-05, atol=1e-06)

    dA2_expected = np.array([[0.12913162,-0.44014127],
                    [-0.14175655,0.48317296],
                    [0.01663708,-0.05670698]])

    assert np.allclose(grads["dA2"], dA2_expected, rtol=1e-05, atol=1e-06)

    print("... end test")


def test_update_parameters():
    print("\nTest update_parameters()...")
    """
    parameters = {'W1': np.array([[ 1.78862847,  0.43650985,  0.09649747],
        [-1.8634927 , -0.2773882 , -0.35475898],
        [-0.08274148, -0.62700068, -0.04381817],
        [-0.47721803, -1.31386475,  0.88462238]]),
        'W2': np.array([[ 0.88131804,  1.70957306,  0.05003364, -0.40467741],
        [-0.54535995, -1.54647732,  0.98236743, -1.10106763],
        [-1.18504653, -0.2056499 ,  1.48614836,  0.23671627]]),
        'W3': np.array([[-1.02378514, -0.7129932 ,  0.62524497],
        [-0.16051336, -0.76883635, -0.23003072]]),
        'b1': np.array([[ 0.],
        [ 0.],
        [ 0.],
        [ 0.]]),
        'b2': np.array([[ 0.],
        [ 0.],
        [ 0.]]),
        'b3': np.array([[ 0.],
        [ 0.]])}
        grads = {'dW1': np.array([[ 0.63070583,  0.66482653,  0.18308507],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]]),
        'dW2': np.array([[ 1.62934255,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ,  0.        ]]),
        'dW3': np.array([[-1.40260776,  0.        ,  0.        ]]),
        'da1': np.array([[ 0.70760786,  0.65063504],
        [ 0.17268975,  0.15878569],
        [ 0.03817582,  0.03510211]]),
        'da2': np.array([[ 0.39561478,  0.36376198],
        [ 0.7674101 ,  0.70562233],
        [ 0.0224596 ,  0.02065127],
        [-0.18165561, -0.16702967]]),
        'da3': np.array([[ 0.44888991,  0.41274769],
        [ 0.31261975,  0.28744927],
        [-0.27414557, -0.25207283]]),
        'db1': 0.75937676204411464,
        'db2': 0.86163759922811056,
        'db3': -0.84161956022334572}
    """

    np.random.seed(2)
    W1 = np.random.randn(3,4)
    b1 = np.random.randn(3,1)
    W2 = np.random.randn(1,3)
    b2 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    np.random.seed(3)
    dW1 = np.random.randn(3,4)
    db1 = np.random.randn(3,1)
    dW2 = np.random.randn(1,3)
    db2 = np.random.randn(1,1)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    parameters = update_parameters(parameters, grads, 0.1)

    print ("W1 = "+ str(parameters["W1"]))
    print ("b1 = "+ str(parameters["b1"]))
    print ("W2 = "+ str(parameters["W2"]))
    print ("b2 = "+ str(parameters["b2"]))

    W1_expected = [
                    [-0.59562069,-0.09991781,-2.14584584,1.82662008],
                    [-1.76569676,-0.80627147,0.51115557,-1.18258802],
                    [-1.0535704,-0.86128581,0.68284052,2.20374577]]
    assert np.allclose(parameters["W1"], W1_expected, rtol=1e-05, atol=1e-06)

    b1_expected = [[-0.04659241],[-1.28888275],[0.53405496]]
    assert np.allclose(parameters["b1"], b1_expected, rtol=1e-05, atol=1e-06)

    W2_expected = [[-0.55569196,0.0354055,1.32964895]]
    assert np.allclose(parameters["W2"], W2_expected, rtol=1e-05, atol=1e-06)

    b2_expected = [[-0.84610769]]
    assert np.allclose(parameters["b2"], b2_expected, rtol=1e-05, atol=1e-06)

    print("... end tests")


if __name__ == "__main__":
    test_linear_forward()
    test_linear_activation_forward()
    test_compute_cost()
    test_forward_propagation()
    test_linear_backward()
    test_linear_activation_backward()
    test_backpropagation()
    test_update_parameters()

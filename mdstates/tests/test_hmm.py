import numpy as np

from ..hmm import (generate_ignore_list, viterbi, viterbi_wrapper,
                   fast_viterbi)
from ..hmm_cython import decoder_cpp


def test_generate_ignore_list():
    mat = np.array([[[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 1], [0, 0]]])
    ignore_list = generate_ignore_list(mat, n=0)
    assert not ignore_list[0], "Ignore list for 0 is not empty."
    assert not ignore_list[1], "Ignore list for 1 is not empty."

    mat = np.array([[[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 1], [0, 0]]])
    ignore_list = generate_ignore_list(mat, n=2)
    assert ignore_list[0] == [[0, 1]],\
        "Ignore list not picking up correct indices."
    assert not ignore_list[1], "Ignore list for 1 is not empty."

    mat = np.array([[[0, 1], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 1], [0, 0]],
                    [[0, 0], [0, 0]],
                    [[0, 0], [0, 0]]])
    ignore_list = generate_ignore_list(mat, n=2)
    assert ignore_list[1] == [[0, 1]],\
        "Ignore list not picking up correct indices."
    assert not ignore_list[0], "Ignore list for 0 is not empty."
    return


def test_viterbi():
    obs = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])
    obs[[75, 80, 83]] = 1
    obs[[128, 141]] = 0

    states = np.array([0, 1])
    start_p = np.array([0.5, 0.5])
    trans_p = np.array([[0.999, 0.001], [0.001, 0.999]])
    emission_p = np.array([[0.6, 0.4], [0.4, 0.6]])

    test = viterbi(obs, states, start_p, trans_p, emission_p)
    assert np.where(np.diff(test) == 1)[0][0] + 1 == 100
    return


def test_viterbi_wrapper():
    obs = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])
    obs[[75, 80, 83]] = 1
    obs[[128, 141]] = 0
    test_inp = (0, 1, obs)

    test_results = viterbi_wrapper(test_inp)
    assert test_results[0] == 0
    assert test_results[1] == 1
    assert np.where(np.diff(test_results[2]) == 1)[0][0] + 1 == 100
    return


def test_fast_viterbi():
    obs = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])
    obs[[75, 80, 83]] = 1
    obs[[128, 141]] = 0

    states = np.array([0, 1])
    start_p = np.array([0.5, 0.5])
    trans_p = np.array([[0.999, 0.001], [0.001, 0.999]])
    emission_p = np.array([[0.6, 0.4], [0.4, 0.6]])

    test = fast_viterbi(obs, states, start_p, trans_p, emission_p)
    assert np.where(np.diff(test) == 1)[0][0] + 1 == 100
    return


def test_decoder_cpp():
    obs = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])
    obs[[75, 80, 83]] = 1
    obs[[128, 141]] = 0

    # states = np.array([0, 1])
    # start_p = np.array([0.5, 0.5])
    # trans_p = np.array([[0.999, 0.001], [0.001, 0.999]])
    # emission_p = np.array([[0.6, 0.4], [0.4, 0.6]])

    test = decoder_cpp(obs)
    assert np.where(np.diff(test) == 1)[0][0] + 1 == 100
    return

from libcpp.vector cimport vector
import numpy as np


cdef extern from "hmm.h":
    vector[int] decoder(vector[int] obs, const int num_frames,
                        float start_p[2], float trans_p[2][2], float emission_p[2][2])


def decoder_cpp(obs_list, start_p, trans_p, emission_p):
    """Viterbi algorithm for decoding noisy signal

    Parameters
    ----------
    obs_list : array-like
    start_p : array-like
        Probability of starting in a particular state.
    trans_p : array-like
        Probability of transitioning from one state to another.
    emission_p : array-like
        Probability of emitting one state given its hidden state.

    Returns
    -------
    list
        Cleaned signal with the highest probability of matching the
        observed data."""

    if type(start_p) is not np.ndarray:
        start_p = np.array(start_p)
    if type(trans_p) is not np.ndarray:
        trans_p = np.array(trans_p)
    if type(emission_p) is not np.ndarray:
        emission_p = np.array(emission_p)
    
    length = len(obs_list)

    cdef float cstart_p[2]
    for i in range(2):
        cstart_p[i] = start_p[i]

    cdef float ctrans_p[2][2]
    for i in range(2):
        for j in range(2):
            ctrans_p[i][j] = trans_p[i, j]

    cdef float cemission_p[2][2]
    for i in range(2):
        for j in range(2):
            cemission_p[i][j] = emission_p[i, j]

    cdef vector[int] obs_vector = obs_list
    return decoder(obs_vector, length, cstart_p, ctrans_p, cemission_p)

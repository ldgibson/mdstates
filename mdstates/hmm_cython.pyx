from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport numpy as np
import numpy as np


cdef extern from "hmm.h":
    void decode(int **obs, const int num_bonds, const int num_frames,
                float start_p[2], float trans_p[2][2], float emission_p[2][2],
                int cores)
    void viterbi(int *obs, const int num_frames, float start_p[2],
                 float trans_p[2][2], float emission_p[2][2])


def decode_cpp(np.ndarray[int, ndim=2] obs_arr, start_p,
               trans_p, emission_p, cores):
    """Viterbi algorithm for decoding noisy signal

    Parameters
    ----------
    obs_arr : np.ndarray
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

    # Change the HMM parameters into float C arrays
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

    # Force the array to be C-contiguous
    # i.e., each row has its own contiguous allocation of memory
    if not obs_arr.flags['C_CONTIGUOUS']:
        obs_arr = np.ascontiguousarray(obs_arr)

    cdef int num_bonds = obs_arr.shape[0]
    cdef int num_frames = obs_arr.shape[1]

    # Create a memoryview of the numpy array
    # Again, forcing the memoryview to be C-contiguous
    cdef int[:, ::1] obs_memview = obs_arr

    # Do not over-allocate resources if there is not enough
    # data to fill them.
    if num_bonds < cores:
        cores = num_bonds

    # Build a pointer to array (which will contain pointers to rows)
    cdef int **point_to_arr = <int **>malloc(num_bonds * sizeof(int*))
    if not point_to_arr: raise MemoryError
    try:
        for i in range(num_bonds):
            point_to_arr[i] = &obs_memview[i, 0]
        decode(&point_to_arr[0], num_bonds, num_frames, cstart_p,
               ctrans_p, cemission_p, cores)
    finally:
        # Deallocate the reserved memory from the pointers
        free(point_to_arr)
    return obs_arr


def viterbi_cpp(np.ndarray[int, ndim=1] obs, start_p, trans_p,
                emission_p):

    # Change the HMM parameters into float C arrays
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

    if not obs.flags['C_CONTIGUOUS']:
        obs = np.ascontiguousarray(obs)

    cdef int num_frames = obs.shape[0]
    cdef int[::1] obs_memview = obs
    viterbi(&obs_memview[0], num_frames, cstart_p, ctrans_p,
            cemission_p)
    return obs

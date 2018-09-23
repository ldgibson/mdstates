from libcpp.vector cimport vector


cdef extern from "hmm.h":
    vector[int] decoder(vector[int] obs, const int num_frames)


def decoder_cpp(obs_list):
    length = len(obs_list)
    cdef vector[int] obs_vector = obs_list
    return decoder(obs_vector, length)

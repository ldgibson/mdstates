from libcpp.vector cimport vector


cdef extern from "hmm.h":
    vector[int] decoder(vector[int] obs)


def decoder_cpp(obs_list):
    cdef vector[int] obs_vector = obs_list
    return decoder(obs_vector)

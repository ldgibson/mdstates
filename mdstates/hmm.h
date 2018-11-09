#ifndef HMM_H
#define HMM_H

void decode(int **obs, const int num_bonds, const int num_frames, float start_p[2], float trans_p[2][2], float emission_p[2][2], int cores);
void viterbi(int *obs, const int num_frames, float start_p[2], float trans_p[2][2], float emission_p[2][2]);

#endif

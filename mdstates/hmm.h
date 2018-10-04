#ifndef DECODER
#define DECODER

std::vector<int> decoder(std::vector<int> obs, const int num_frames, 
                         float start_p[2], float trans_p[2][2], float emission_p[2][2]);

#endif

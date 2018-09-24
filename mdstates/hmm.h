#ifndef DECODER
#define DECODER

std::vector<int> decoder(std::vector<int> obs, const int num_frames, 
                         std::vector<float> start_p,
                         std::vector<std::vector<float>> trans_p,
                         std::vector<std::vector<float>> emission_p);

#endif

#include <stdio.h>
#include <math.h>
#include <vector>


float max(float x, float y) {
    if (x > y) {
        return(x);
    } else {
        return(y);
    }
}


int main() {
    int states[2] = { 0, 1 };
    float start_p[2] = { 0.5, 0.5 };
    float trans_p[2][2] = { { 0.999, 0.001 },
                            { 0.001, 0.999 } };
    float emission_p[2][2] = { { 0.60, 0.40 },
                               { 0.40, 0.60 } };

    int obs[1000] = {};
    int x;
    for (x = 0; x<1000; x++) {
        if (x < 300) {
            obs[x] = 0;
        } else if ((x < 400) && (x % 3 == 0)){
            obs[x] = 0;
        } else {
            obs[x] = 1;
        }
        printf("x = %d, obs = %d\n", x, obs[x]);
    }
    // obs[280] = 1;
    // obs[276] = 1;
    // obs[250] = 1;

    int i, j;
    for ( i = 0; i < 2; i++ ) {
        printf("State: %d\n", states[i]);
        printf("Start Probability: %f\n", start_p[i]);
        for ( j = 0; j < 2; j++ ) {
            printf("Transition Probability: %f\n", trans_p[i][j]);
            printf("Emission Probability: %f\n", emission_p[i][j]);
        }
    }

    // 1000 frames, 2 states, log prob and prev value for each.
    float V[1000][2] = {{}};
    int prev[1000][2] = {{}};

    for (i = 0; i < 2; i++) {
            V[0][i] = start_p[i] * emission_p[i][obs[i]];
            prev[0][i] = -1;
            // printf("Start probability for state %d: %f\n", i, V[0][i]);
            // printf("Previous state for state %d: %d\n", i, prev[0][i]);
    }

    int t, prev_st, st;
    float max_tr_prob[2];
    float max_tr, max_prob;
    for (t = 1; t < 1000; t++) {
        for (st = 0; st < 2; st++) {
            for (prev_st = 0; prev_st < 2; prev_st++) {
                max_tr_prob[prev_st] = V[t - 1][prev_st] + log10(trans_p[prev_st][st]);
            }
            max_tr = max(max_tr_prob[0], max_tr_prob[1]);
            for (prev_st = 0; prev_st < 2; prev_st++) {
                if (V[t - 1][prev_st] + log10(trans_p[prev_st][st]) == max_tr) {
                    max_prob = max_tr + log10(emission_p[st][obs[t]]);
                    V[t][st] = max_prob;
                    prev[t][st] = prev_st;
                    break;
                }
            }

        }
        // printf("At t = %d, State 0 prob = %f, State 1 prob = %f\n", t, V[t][0], V[t][1]);
    }
    std::vector <int> optimal_path;
    float max_final_prob = max(V[999][0], V[999][1]);
    int previous;

    for (st = 0; st < 2; st++) {
        if (V[999][st] == max_final_prob) {
            optimal_path.push_back(st);
            previous = st;
        }
    }
    printf("I got here!\n");

    for (t = 1000 - 2; t >= 0; t--) {
        optimal_path.insert(optimal_path.begin(), prev[t + 1][previous]);
        previous = prev[t + 1][previous];
    }

    for (t = 0; t < 1000; t++) {
        printf("t = %d, state = %d\n", t, optimal_path[t]);
    }

    return(0);
}

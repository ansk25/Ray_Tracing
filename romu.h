#ifndef PROJECT_2_WINTER_2026_AGNES02_GIF_ROMU_H
#define PROJECT_2_WINTER_2026_AGNES02_GIF_ROMU_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t xState;
    uint64_t yState;
    uint64_t zState;
}romu_state;

void split64_seed(uint64_t seed);

uint64_t next(uint64_t seed);

uint64_t romuTrio_random(romu_state *s);

void romuTrio_seed(romu_state *s, uint64_t seed);

// extern inline double romuTrio_double(void);

#ifdef __cplusplus
}
#endif

#endif
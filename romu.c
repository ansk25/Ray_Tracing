#include <stdint.h>
#include "romu.h"
/* This is a fixed-increment version of Java 8's SplittableRandom generator
   See http://dx.doi.org/10.1145/2714064.2660195 and
   http://docs.oracle.com/javase/8/docs/api/java/util/SplittableRandom.html

   It is a very fast generator passing BigCrush, and it can be useful if
   for some reason you absolutely want 64 bits of state. */

// static uint64_t x; /* The state can be seeded with any value. */

uint64_t next(uint64_t seed) {
    uint64_t z = (seed += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Romu Pseudorandom Number Generators
//
// Copyright 2020 Mark A. Overton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ------------------------------------------------------------------------------------------------
//
// Website: romu-random.org
// Paper:   http://arxiv.org/abs/2002.11331
//
// Copy and paste the generator you want from those below.
// To compile, you will need to #include <stdint.h> and use the ROTL definition below.

#define ROTL(d,lrot) ((d<<(lrot)) | (d>>(8*sizeof(d)-(lrot))))

// typedef struct {
//     uint64_t xState;
//     uint64_t yState;
//     uint64_t zState;
// }romu_state;

inline uint64_t romuTrio_random(romu_state *s)
{
    const uint64_t xp = s->xState;
    const uint64_t yp = s->yState;
    const uint64_t zp = s->zState;
    s->xState = 15241094284759029579u * zp;
    s->yState = yp - xp;  s->yState = ROTL(s->yState, 12);
    s->zState = zp - yp;  s->zState = ROTL(s->zState, 44);
    return xp;
}

void romuTrio_seed(romu_state *s, uint64_t seed)
{
    s->xState = next(seed);
    s->yState = next(s->xState);
    s->zState = next(s->yState);
}

// inline double romuTrio_double(void)
// {
//     return (romuTrio_random() >> 11) * INV_2_53;
// }




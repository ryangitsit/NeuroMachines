
#ifndef _BRIAN_OBJECTS_H
#define _BRIAN_OBJECTS_H

#include "synapses_classes.h"
#include "brianlib/clocks.h"
#include "brianlib/dynamic_array.h"
#include "brianlib/stdint_compat.h"
#include "network.h"
#include "randomkit.h"
#include<vector>


namespace brian {

// In OpenMP we need one state per thread
extern std::vector< rk_state* > _mersenne_twister_states;

//////////////// clocks ///////////////////
extern Clock defaultclock;

//////////////// networks /////////////////
extern Network network;

//////////////// dynamic arrays ///////////
extern std::vector<int32_t> _dynamic_array_spikemonitor_i;
extern std::vector<double> _dynamic_array_spikemonitor_t;

//////////////// arrays ///////////////////
extern double *_array_defaultclock_dt;
extern const int _num__array_defaultclock_dt;
extern double *_array_defaultclock_t;
extern const int _num__array_defaultclock_t;
extern int64_t *_array_defaultclock_timestep;
extern const int _num__array_defaultclock_timestep;
extern int32_t *_array_poissongroup__spikespace;
extern const int _num__array_poissongroup__spikespace;
extern int32_t *_array_poissongroup_i;
extern const int _num__array_poissongroup_i;
extern double *_array_poissongroup_rates;
extern const int _num__array_poissongroup_rates;
extern int32_t *_array_spikemonitor__source_idx;
extern const int _num__array_spikemonitor__source_idx;
extern int32_t *_array_spikemonitor_count;
extern const int _num__array_spikemonitor_count;
extern int32_t *_array_spikemonitor_N;
extern const int _num__array_spikemonitor_N;

//////////////// dynamic arrays 2d /////////

/////////////// static arrays /////////////

//////////////// synapses /////////////////

// Profiling information for each code object
}

void _init_arrays();
void _load_arrays();
void _write_arrays();
void _dealloc_arrays();

#endif



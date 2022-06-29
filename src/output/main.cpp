#include <stdlib.h>
#include "objects.h"
#include <ctime>
#include <time.h>

#include "run.h"
#include "brianlib/common_math.h"
#include "randomkit.h"

#include "code_objects/poissongroup_thresholder_codeobject.h"
#include "code_objects/after_run_poissongroup_thresholder_codeobject.h"
#include "code_objects/spikemonitor_codeobject.h"


#include <iostream>
#include <fstream>
#include <string>




int main(int argc, char **argv)
{
        

	brian_start();
        

	{
		using namespace brian;

		
                
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_defaultclock_dt[0] = 0.0001;
        _array_poissongroup_rates[0] = 16.0;
        _array_defaultclock_timestep[0] = 0;
        _array_defaultclock_t[0] = 0.0;
        network.clear();
        network.add(&defaultclock, _run_poissongroup_thresholder_codeobject);
        network.add(&defaultclock, _run_spikemonitor_codeobject);
        network.run(0.5, NULL, 10.0);
        _after_run_poissongroup_thresholder_codeobject();
        #ifdef DEBUG
        _debugmsg_spikemonitor_codeobject();
        #endif

	}
        

	brian_end();
        

	return 0;
}
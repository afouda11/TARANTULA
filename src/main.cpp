/*
TARANTULA
Copyright (C) 2022  Adam E. A. Fouda
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#define _USE_MATH_DEFINES
#include "read_and_write.h"
#include "xfel_tdse.h"
#include "spawn_decay.h"

#include <armadillo>
#include <iostream>
#include <iterator>
#include <fstream>
#include <math.h>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <sstream>
#include <map>

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
    //vector<bool> * BOOL_VEC = new vector<bool>(16);
	string calc_type;
	read_options("CALC_TYPE", calc_type);

    if (calc_type == "TDSE_XFEL") {
		XFEL_TDSE();
	}

    if (calc_type == "SPAWN_DECAY") {
		SPAWN_DECAY();
	}

}



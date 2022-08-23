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
#ifndef rk4_hpp
#define rk4_hpp

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <complex>
#include "vectypedef.hpp"
#include <armadillo>
using namespace std;


//typedef complex<double>                         complexd;
//typedef vector<complexd>                        vec1x;
//typedef vector<vec1x>                           vec2x;

class EOMDRIVER {

public:

	EOMDRIVER();
	~EOMDRIVER();

	int n;
	vector<arma::mat> Matrix;
	vector<vector<double> > mu;
	vector<double> Et;
	vector<double> wx;
	vector<vector<double> > auger_gamma;
	vector<vector<double> > photo_gamma;
	vector<string> decay_channels;
	vector<bool> BOOL_VEC;
	int n_pulse;

void RK4(vec1x & y, double t0, double tf);

double Analytical_Population_Loss(double tf, int j, int k);
//double Numerical_Population_Loss(int i, int j, int k, double dt, vec1x pt);
void Numerical_Population_Loss(int i, int j, int k, double dt, int nt, complex<double> pt_state, complex<double> & pt_loss, complex<double> pt_loss_prev);

private:

void REQ(double t, vec1x y, vec1x & dydt);

double Stark_Shift(int state, vector<double> auger_i, vector<double> photo_i, int n);

void Dipole_Matrix_Element(double & dipole_mat, vector<double> mu, int i, int j);

};

#endif /* rk4_hpp */

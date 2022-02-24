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

EOMDRIVER(void);

	int neqn;
	vector<arma::mat> Matrix;
	vector<vector<double> > mu;
	vector<double> Et;
	vector<double> wx;
	vector<vector<vector<double> > > decay_widths;
	vector<string> decay_channels;
	vector<bool> BOOL_VEC;

void RK4(vec1x & y, double t0, double tf);

private:

void REQ(double t, vec1x y, vec1x & dydt);

double Stark_Shift(int state, vector<double> auger_i, vector<double> photo_i, int n);

void Dipole_Matrix_Element(double & dipole_mat, vector<double> mu, int i, int j);

};

#endif /* rk4_hpp */

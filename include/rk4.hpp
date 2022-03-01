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

	int n;
	vector<arma::mat> Matrix;
	vector<vector<double> > mu;
	vector<double> Et;
	vector<double> wx;
	vector<vector<double> > auger_gamma;
	vector<vector<double> > photo_gamma;
	vector<vector<vector<double> > > photo_gamma_vec;
	vector<string> decay_channels;
	vector<bool> BOOL_VEC;

void RK4(vec1x & y, double t0, double tf);

double Analytical_Population_Loss(double tf, int j, int k);
double Numerical_Population_Loss(int i, int j, int k, double dt, vec1x pt);

private:

void REQ(double t, vec1x y, vec1x & dydt);

double Stark_Shift(int state, vector<double> auger_i, vector<double> photo_i, int n);

void Dipole_Matrix_Element(double & dipole_mat, vector<double> mu, int i, int j);

};

#endif /* rk4_hpp */

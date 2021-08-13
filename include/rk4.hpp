//
//  rk4.hpp
//  
//
// 
//

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

void rk4(int neqn, vec1x & y, double t0, double tf, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, vector<string>
decay_channels, bool RWA, bool DECAY, bool TWOPULSE, bool STARK, bool DECAY_AMP);

void REQ(double t, int neqn, vec1x y, vec1x & dydt, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, vector<string>
decay_channels, bool RWA, bool DECAY, bool TWOPULSE, bool STARK, bool DECAY_AMP);

double Stark_Shift(int state, double Et, double gamma_state, vector<double> auger_i, vector<double> photo_i, int n, double wx, arma::mat Matrix, vector<double> polarization);
#endif /* rk4_hpp */

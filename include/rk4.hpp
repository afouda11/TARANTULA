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

void rk4_c(int neqn, vector < complex<double> > & y, double t0, double tf);
void rk4_d(int neqn, vector < double > & y, double t0, double tf);
void REQ_d(double t, int n, vector <double> y, vector <double> & dydt);
void REQ_c(double t, int n, vector <complex <double> > y, vector <complex <double> > & dydt);

void rk4(int neqn, vec1x & y, double t0, double tf, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, bool RWA, bool DECAY, bool TWOPULSE, bool STARK);

void REQ(double t, int n, vec1x y, vec1x & dydt, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, bool RWA, bool DECAY, bool TWOPULSE, bool STARK);

double Stark_Shift(int state, double Et, double gamma_state, vector<double> auger_i, vector<double> photo_i, int n, double wx, arma::mat Matrix, vector<double> polarization);
#endif /* rk4_hpp */

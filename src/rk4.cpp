
//  rk4.cpp
//  
//
//  rk4 is a time propagator for solving dy/dt = f(y,t)
//  REQ computes f(y,t)
//
#include <cmath>
#include "rk4.hpp"
#include <armadillo>
#include <complex>

#include "read_and_write.h"

#ifdef _OPENMP
#include <omp.h>
#endif

EOMDRIVER::EOMDRIVER(void){

}

void EOMDRIVER::RK4(vec1x & y, double t0, double tf) {
  
  double dt = tf - t0;
  double tc = t0 + 0.5* dt;
  double dt2 = 0.5*dt;
  double dt6 = dt/6.0;
  
  // define and initialize working arrays
  vec1x  k1 (n, complexd (0,0) );
  vec1x  k2 (n, complexd (0,0) );
  vec1x  k3 (n, complexd (0,0) );
  vec1x  k4 (n, complexd (0,0) );
  vec1x  ytemp (n, complexd (0,0) );
  
  // step k1
  REQ(t0, y, k1);
  
  for (int i = 0; i< n; i++){
    ytemp[i] = y[i] + dt2*k1[i];
  }
  
  // step k2
  REQ(tc, ytemp, k2);
  
  for (int i = 0; i< n; i++){
    ytemp[i] = y[i] + dt2*k2[i];
  }
  
  // step k3
  REQ(tc, ytemp, k3);
  
  for (int i = 0; i< n; i++){
    ytemp[i] = y[i] + dt*k3[i];
  }
  
  // step k4
  REQ(tf, ytemp, k4);
  
  for (int i = 0; i< n; i++){
    y[i] = y[i] + dt6*(k1[i] + 2.0*k2[i]+ 2.0*k3[i]+ k4[i]);
  }
  
  return;
}

/*
 solution:
*/
void EOMDRIVER::REQ(double t, vec1x y, vec1x & dydt){

  	const complex<double> I(0,1);

    double R = 0.0;

    if(!BOOL_VEC[5]) { //1 pulse
	    //DIAGONAL	
	    for (int j = 0; j < n; j++) {
            if(BOOL_VEC[10]) {
                R = Stark_Shift(j, auger_gamma[0], photo_gamma[0], n);
                dydt[j] = ( Matrix[0](j,j) - ( I * ( ((auger_gamma[0][j] + photo_gamma[0][j]) /2.0) + (I * R) ) ) ) * y[j];
            }
            else {
                //dydt[j] = ( ( Matrix[0](j,j) ) - (( I * (auger_gamma[0][j] + photo_gamma[0][j]) )/2.0) ) * y[j];
                dydt[j] = ( ( Matrix[0](j,j) ) - (( I * (auger_gamma[0][j] + photo_gamma[0][j]) )/2.0) ) * y[j];
            }     
	        //OFF DIAGONAL
		    for (int i = 0;  i < n; i++) {
	            double dipole_mat = 0.0;
				Dipole_Matrix_Element(dipole_mat ,mu[0], i, j); 
			    if (BOOL_VEC[0]) {//RWA
				    if (i != j) {
					    if (j > i ) {
							if (!BOOL_VEC[9]) {//NO TWO STATE, 
						    	dydt[j] += ((((dipole_mat * Et[0] * exp(-1.0 * I * wx[0] * t))) / 2.0) * y[i]);
							}
							if (BOOL_VEC[9]) {
						    	dydt[j] += ((exp(-1.0 * I * t)) * y[i]);
							}
					    }
					    if (j < i ) {
							if (!BOOL_VEC[9]) {
						    	dydt[j] += ((((dipole_mat * Et[0] * exp(1.0 * I * wx[0] * t))) / 2.0) * y[i]);
							}
							if (BOOL_VEC[9]) {
						    	dydt[j] += ((exp(1.0 * I * t)) * y[i]);
							}
					    }
				    }
 			    }
			    if (!BOOL_VEC[0]) {
				    if (i != j) {
					    dydt[j] += (dipole_mat * Et[0] * cos(wx[0] * t)) * y[i];	
				    }
			    }
		    }	 
		    dydt[j] /= I;		
	    }
    }
    if (BOOL_VEC[5]) {//2 pulse 
        for (int j = 0; j < n; j++) {//check how the augers treated for two different core hole lifetimes
				complex<double> d1 = 0.0;
				d1 = (I *(((auger_gamma[0][j]+auger_gamma[1][j])*0.5) + photo_gamma[0][j]+photo_gamma[1][j]))/2.0;
                dydt[j] = (Matrix[0](j,j) - d1) * y[j];
	        //OFF DIAGONAL
            for (int i = 0;  i < n; i++) {
				complex<double> offd1 = 0.0;
				complex<double> offd2 = 0.0;
	            vector<double> dipole_mat(2);
				Dipole_Matrix_Element(dipole_mat[0], mu[0], i, j);
				Dipole_Matrix_Element(dipole_mat[1], mu[1], i, j);
                //dipole_mat[0]=(Matrix[0](i,j)*mu[0][0])+(Matrix[1](i,j)*mu[0][1])+(Matrix[2](i,j)*mu[0][2]);
                //dipole_mat[1]=(Matrix[0](i,j)*mu[1][0])+(Matrix[1](i,j)*mu[1][1])+(Matrix[2](i,j)*mu[1][2]); 
			    if (BOOL_VEC[0]) { //RWA
				    if (i != j) {
					    if (j > i ) {
							offd1 = dipole_mat[0] * Et[0] * exp(-1.0 * I * wx[0] * t);
							offd2 = dipole_mat[1] * Et[1] * exp(-1.0 * I * wx[1] * t);
						    dydt[j] += (((offd2 + offd2) / 2.0) * y[i]);
					    }
					    if (j < i ) {
							offd1 = (dipole_mat[0] * Et[0] * exp( 1.0 * I * wx[0] * t));
							offd2 = (dipole_mat[1] * Et[1] * exp( 1.0 * I * wx[1] * t));
						    dydt[j] += (((offd1 + offd2) / 2.0) * y[i]);
					    }
				    }
 			    }
			    if (!BOOL_VEC[0]) { //No RWA
				    if (i != j) {
						offd1 = (dipole_mat[0] * Et[0] * cos(wx[0] * t));
						offd2 = (dipole_mat[1] * Et[1] * cos(wx[1] * t));
					    dydt[j] += (offd1 + offd2) * y[i];	
				    }
			    }
		    }
	        dydt[j] /= I;		
        }
    }
  /*  if(BOOL_VEC[13] && !BOOL_VEC[5]) { //Decay amp and 1 pulse

        for(int k = 0; k < n_decay_chan; k++) {

        	for (int j = (n * (k+1)); j < n * (k+2); j++) {

				dydt[j]  = 1.0 * y[j] * Matrix[0](j-(n*(k+1)),j-(n*(k+1)));

				if (decay_channels[k] == "PHOTO_TOTAL" ) {
					dydt[j] += (y[j-(n*(k+1))] * pow((photo_gamma[0][j-(n*(k+1))] / (2*M_PI)),4.0));
					//dydt[j] += (y[j] * pow((photo_gamma[0][j-(n*(k+1))] / (2*M_PI)),0.5));
				}
				if (decay_channels[k] == "AUGER" ) {
					dydt[j] += (y[j-(n*(k+1))] * pow((auger_gamma[0][j-(n*(k+1))] / (2)),1.0));
					//dydt[j] += (y[j] * pow((auger_gamma[0][j-(n*(k+1))] / (2*M_PI)),0.5));
				}

            	for (int i = (n * (k+1)); i < n * (k+2); i++) {
				    if (i != j) {
						if (decay_channels[k] == "PHOTO_TOTAL" ) {
							dydt[j] += (y[j-(n*(k+1))] * pow((photo_gamma[0][j-(n*(k+1))] / (2*M_PI)),0.5));
						}
						if (decay_channels[k] == "AUGER" ) {
							dydt[j] += (y[j-(n*(k+1))] * pow((auger_gamma[0][j-(n*(k+1))] / (2*M_PI)),0.5));
						}
					}	
				}
                dydt[j] /= I;
            }
        }        
    }*/

    return;

}

double EOMDRIVER::Analytical_Population_Loss(double tf, int j, int k)
{
	double gamma = 0.0;
	if (decay_channels[k] == "AUGER" ) {
		gamma = auger_gamma[0][j-(n*(k+1))];
	}
	if (decay_channels[k] == "PHOTO_TOTAL" ) {
		gamma = photo_gamma[0][j-(n*(k+1))];
	}

	double beta = -16.0 + pow(gamma,2.0);
	complex<double> sqrtbeta = sqrt(complex<double>(beta, 0.0));

	double prod1   = pow(gamma,2.0)*(cosh((sqrtbeta*tf)/2.0).real());
	double prod2   = gamma*((sqrtbeta*sinh((sqrtbeta*tf)/2.0)).real());
	double expterm = exp((-1*gamma*tf)/2.0);

	return (1-((expterm*(-16+prod1+prod2))/beta));
}

double EOMDRIVER::Numerical_Population_Loss(int i, int j, int k, double dt, vec1x pt)
{
	double dum = 0.0;
	//Use Trapezoidal Rule
/*	for (int step = 0; step < i; step++) {
	
		//cout << pt[i].real() << endl;	
		if (step == 0) {
			dum += pt[step].real();
		}
		else if (step == i) {
			dum += pt[step].real();
		}
		else {
			dum += (2.0 * pt[step].real());
		}
	}
	dum *= (dt/2.0);*/

	//Simpson 1/3 Rule
	for (int step = 0; step <= i; step++) {
		if (step == 0 || step == i) {
			dum += pt[step].real();
		}
		else if (step % 2 != 0) {	
			dum += (4 * pt[step].real());
		}
		else {
			dum += (2 * pt[step].real());
		}
	}
	dum *= (dt /3);
	dum *= auger_gamma[0][j-(n*(k+1))];	

	return dum;

}

double EOMDRIVER::Stark_Shift(int state, vector<double> auger_i, vector<double> photo_i, int n) 
{
    double R = 0.0;
	double gamma_state = auger_i[state] + photo_i[state]; 
    for (int i = 0;  i < n; i++) {

        double dipole_mat = ((Matrix[0](i,state)*mu[0][0])+(Matrix[0](i,state)*mu[0][1])+(Matrix[0](i,state)*mu[0][2]));
        double w_statei = Matrix[0](i,i) - Matrix[0](state,state); 

        R = R + (( pow(dipole_mat, 2) * w_statei ) / ( pow(w_statei-wx[0], 2) + pow((auger_i[i] + photo_i[i])/ 2, 2) ) ); 
    }

    R = R * ( ( pow(Et[0], 2) * gamma_state ) / 4);   

    return R;
}

void EOMDRIVER::Dipole_Matrix_Element(double & dipole_mat, vector<double> mu, int i, int j)
{
	dipole_mat=(Matrix[0](i,j)*mu[0])+(Matrix[1](i,j)*mu[1])+(Matrix[2](i,j)*mu[2]);	
}


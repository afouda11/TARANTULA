
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

void rk4(int neqn, int nval, vec1x & y, double t0, double tf, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, bool RWA, bool DECAY, bool TWOPULSE, bool STARK){
  
  double dt = tf - t0;
  double tc = t0 + 0.5* dt;
  double dt2 = 0.5*dt;
  double dt6 = dt/6.0;
  
  // define and initialize working arrays
  vec1x  k1 (neqn, complexd (0,0) );
  vec1x  k2 (neqn, complexd (0,0) );
  vec1x  k3 (neqn, complexd (0,0) );
  vec1x  k4 (neqn, complexd (0,0) );
  vec1x  ytemp (neqn, complexd (0,0) );
  
  // step k1
  REQ(t0,neqn,nval,y,k1,Matrix, polarization, Et, wx, decay_widths, RWA, DECAY, TWOPULSE, STARK);
  
  for (int i = 0; i< neqn; i++){
    ytemp[i] = y[i] + dt2*k1[i];
  }
  
  // step k2
  REQ(tc,neqn,nval,ytemp,k2,Matrix, polarization, Et, wx, decay_widths, RWA, DECAY, TWOPULSE, STARK);
  
  for (int i = 0; i< neqn; i++){
    ytemp[i] = y[i] + dt2*k2[i];
  }
  
  // step k3
  REQ(tc,neqn,nval,ytemp,k3,Matrix, polarization, Et, wx, decay_widths, RWA, DECAY, TWOPULSE, STARK);
  
  for (int i = 0; i< neqn; i++){
    ytemp[i] = y[i] + dt*k3[i];
  }
  
  // step k4
  REQ(tf,neqn,nval,ytemp,k4,Matrix, polarization, Et, wx, decay_widths, RWA, DECAY, TWOPULSE, STARK);
  
  for (int i = 0; i< neqn; i++){
    y[i] = y[i] + dt6*(k1[i] + 2.0*k2[i]+ 2.0*k3[i]+ k4[i]);
  }
  
  return;
}

/*
 solution:
*/
void REQ(double t, int n, int nval, vec1x y, vec1x & dydt, vector<arma::mat> Matrix, vector<vector<double> >
polarization, vector<double> Et, vector<double> wx, vector<vector<vector<double> > > decay_widths, bool RWA, bool DECAY, bool TWOPULSE, bool STARK){
  
  	const complex<double> I(0,1);
    vector<vector<double> > auger_gamma;
    vector<vector<double> > photo_gamma;
    vector<vector<double> > photo_sigma;

    if (DECAY) {
        
        if(!TWOPULSE) {
            auger_gamma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][0].size()), 0.0));
            photo_sigma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            photo_gamma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            for(int i = 0; i < static_cast<int>(decay_widths[0][0].size()); i++) {
                auger_gamma[0][i] = decay_widths[0][0][i] / 27.2114;
                photo_sigma[0][i] = decay_widths[0][1][i] / 28.0175; //convert megabarn to a.u.
                photo_gamma[0][i] = (photo_sigma[0][i] / wx[0]) * pow(Et[0],2);
            }       
        }      
        if(TWOPULSE) {
            auger_gamma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][0].size()), 0.0));
            photo_sigma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            photo_gamma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            for(int i = 0; i < static_cast<int>(decay_widths[0][0].size()); i++) {

                auger_gamma[0][i] = decay_widths[0][0][i] / 27.2114;
                auger_gamma[1][i] = decay_widths[1][0][i] / 27.2114;

                photo_sigma[0][i] = decay_widths[0][1][i] / 28.0175;
                photo_sigma[1][i] = decay_widths[1][1][i] / 28.0175;

                photo_gamma[0][i] = (photo_sigma[0][i] / wx[0]) * pow(Et[0],2);
                photo_gamma[1][i] = (photo_sigma[1][i] / wx[1]) * pow(Et[1],2);

            }        
        }      
        
    }
    if (!DECAY) {
        if(!TWOPULSE) {
            auger_gamma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][0].size()), 0.0));
            photo_sigma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            photo_gamma = vector<vector<double> > (1, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
        }     
        if(TWOPULSE) {
            auger_gamma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][0].size()), 0.0));
            photo_sigma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
            photo_gamma = vector<vector<double> > (2, vector<double>(static_cast<int>(decay_widths[0][1].size()), 0.0));
        }     
    }      
    double R = 0.0;
    if(!TWOPULSE) {
	    //DIAGONAL	
	    for (int j = 0; j < n; j++) {
            if (j >= nval ){ //intermediate states
                if(STARK) {
                    R = Stark_Shift(j, Et[0], auger_gamma[0][2] + photo_gamma[0][2], auger_gamma[0][1] +
                    photo_gamma[0][1], n, wx[0], Matrix[0], polarization[0]);
                    dydt[j] = ( Matrix[0](j,j) - ( I * ( ((auger_gamma[0][2] + photo_gamma[0][2]) /2.0) + (I * R) ) ) ) * y[j];
                }
                else {
                    dydt[j] = ( ( Matrix[0](j,j) ) - (( I * (auger_gamma[0][2] + photo_gamma[0][2]) )/2.0) ) * y[j];
                }     
            }
            else if (j == 0) { //ground state 
                if(STARK) {
                    R = Stark_Shift(j, Et[0], auger_gamma[0][0] + photo_gamma[0][0], photo_gamma[0][1] + auger_gamma[0][1], n, wx[0], Matrix[0], polarization[0]);
                }
                else {
                    dydt[j] = ( ( Matrix[0](j,j) ) - (( I * (auger_gamma[0][0] + photo_gamma[0][0]) )/2.0) ) * y[j];
                }     
            }      
            else {
                if(STARK) { //final states
                    R = Stark_Shift(j, Et[0], photo_gamma[0][1] + auger_gamma[0][1], auger_gamma[0][2] +
                    photo_gamma[0][2], n, wx[0], Matrix[0], polarization[0]);
                    dydt[j] = ( Matrix[0](j,j) - ( I * ( (photo_gamma[0][1]/2.0) + (I * R) ) )  ) * y[j];
                }
                else {
                    dydt[j] = ( ( Matrix[0](j,j) ) - (( I * (photo_gamma[0][1] + auger_gamma[0][1]) )/2.0) ) * y[j];
                }     
            }
	        //OFF DIAGONAL
		    for (int i = 0;  i < n; i++) {
	            double dipole_mat_element = ( (Matrix[0](i,j) * polarization[0][0]) + (Matrix[1](i,j) * polarization[0][1]) + (Matrix[2](i,j) * polarization[0][2]) ); 
			    if (RWA) {
				    if (i != j) {
					    if (j > i ) {
						    dydt[j] += (( ((dipole_mat_element * Et[0] * exp(-1.0 * I * wx[0] * t))) / 2.0) * y[i]);
					    }
					    if (j < i ) {
						    dydt[j] += (( ((dipole_mat_element * Et[0] * exp(1.0 * I * wx[0] * t))) / 2.0) * y[i]);
					    }
				    }
 			    }
			    if (!RWA) {
				    if (i != j) {
					    dydt[j] += (dipole_mat_element * Et[0] * cos(wx[0] * t)) * y[i];	
				    }
			    }
		    }	 
		    dydt[j] /= I;		
	    }
    }
    if(TWOPULSE) {
	    //DIAGONAL	
        for (int j = 0; j < n; j++) {
		    if (j >= nval ){ //only auger decay on the core excited states
                dydt[j] = ( ( Matrix[0](j,j) ) - ((I * (((auger_gamma[0][2] + auger_gamma[1][2]) * 0.5)+
                photo_gamma[0][2] + photo_gamma[1][2]) )/2.0) ) * y[j];
            }   
            else if (j == 0) {
                dydt[j] = ( ( Matrix[0](j,j) ) - ((I * (((auger_gamma[0][0] + auger_gamma[1][0]) * 0.5)+
                photo_gamma[0][0] + photo_gamma[1][0]) )/2.0) ) * y[j];
                
            }          
            else {
                dydt[j] = ( ( Matrix[0](j,j) ) - ((I * (((auger_gamma[0][1] + auger_gamma[1][1]) * 0.5)+
                photo_gamma[0][1] + photo_gamma[1][1]) )/2.0) ) * y[j];
            }
	        //OFF DIAGONAL
            for (int i = 0;  i < n; i++) {
	            vector<double> dipole_mat_element(2); 
                dipole_mat_element[0] = ( (Matrix[0](i,j) * polarization[0][0]) + (Matrix[1](i,j) * polarization[0][1]) + (Matrix[2](i,j) * polarization[0][2]) );
                dipole_mat_element[1] = ( (Matrix[0](i,j) * polarization[1][0]) + (Matrix[1](i,j) * polarization[1][1]) + (Matrix[2](i,j) * polarization[1][2]) ); 
			    if (RWA) {
				    if (i != j) {
					    if (j > i ) {
						    dydt[j] += ( ( ((dipole_mat_element[0] * Et[0] * exp(-1.0 * I * wx[0] * t)) + (dipole_mat_element[1] * Et[1] * exp(-1.0 * I * wx[1] * t))) / 2.0) * y[i]);
					    }
					    if (j < i ) {
						    dydt[j] += ( ( ((dipole_mat_element[0] * Et[0] * exp( 1.0 * I * wx[0] * t)) + (dipole_mat_element[1] * Et[1] * exp( 1.0 * I * wx[1] * t))) / 2.0) * y[i]);
					    }
				    }
 			    }
			    if (!RWA) {
				    if (i != j) {
					    dydt[j] += ( (dipole_mat_element[0] * Et[0] * cos(wx[0] * t)) + (dipole_mat_element[1] * Et[1] * cos(wx[1] * t)) ) * y[i];	
				    }
			    }
		    }
	        dydt[j] /= I;		
        }
    }      

  return;

}

double Stark_Shift(int state, double Et, double gamma_state, double gamma_i, int n, double wx, arma::mat Matrix, vector<double> polarization) {

    double R = 0.0;
    for (int i = 0;  i < n; i++) {

        double dipole_mat_element = ( (Matrix(i,state) * polarization[0]) + (Matrix(i,state) * polarization[1]) + (Matrix(i,state) * polarization[2]) );
        double w_statei = Matrix(i,i) - Matrix(state,state); 

        R = R + (( pow(dipole_mat_element, 2) * w_statei ) / ( pow(w_statei-wx, 2) + pow(gamma_i / 2, 2) ) ); 
        
    }

    R = R * ( ( pow(Et, 2) * gamma_state ) / 4);   

    return R;
}




#define _USE_MATH_DEFINES
#include "read_and_write.h"
#include "pulse_interaction.h"
#include "rk4.hpp"
#include "vectypedef.hpp"

#include <armadillo>
#include <complex>
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

#ifdef _OPENMP
#include <omp.h>
#endif


double gaussian_field(double t, double t_max, double amp_max, double var)
{
   	double Et = amp_max * exp(-1 * (pow((t - t_max),2)) / (2 * (pow(var,2))));

	return Et;
}

void focal_volume_average(double spot_size, vector<vector<double> >& field_strength, vector<double> intensity, int shell_sample)
{
    if (shell_sample > 1) { 
        cout << "Sampling " << intensity.size() << " pulse intensities." << endl;
        cout << "for each intensity the volume avergaing effect from the " << spot_size << " micron^2 focal spot size wiil sample " << shell_sample << " intensity shells.\n" << endl;
    }

    spot_size  = (spot_size * 18897.161646321) / (2 * pow(2 * log(2), 0.5));
    spot_size  = pow(spot_size,2);
    double rs;
    for(int i = 0; i < static_cast<int>(intensity.size()); i++) {
	    for(int j = 0; j < static_cast<int>(field_strength[i].size()); j++) {
            if ( j == 0 ) {
                rs  = j * spot_size;

            }
            else {
                rs = j * spot_size;
            }   
            field_strength[i][j]  =  (intensity[i]/ 3.50944758E+16) * exp( (-1 *rs)/ (2 * spot_size) );
            field_strength[i][j]  = pow(field_strength[i][j], 0.5);
        }
    }
    return;
}

void bandwidth_average(double bw, std::vector<vector<double> >& gw, std::vector<vector<double> >& wn, std::vector<double> wx, int band_sample)
{
    cout << "Sampling " << wx.size() << " central photon energies." << endl;
    cout << "for each central photon energy the " << bw * 27.2114 << " eV " << "bandwith effect will sample " << band_sample << " energies.\n" << endl;
    for(int i = 0; i < static_cast<int>(wx.size()); i++) {
        double step = ((wx[i] + (3 * bw)) - (wx[i] - (3 * bw))) / band_sample;
        for(int j = 0; j < static_cast<int>(gw[i].size()); j++) {
             wn[i][j] = (wx[i] - (3 * bw)) + (j * step);
             gw[i][j] = exp( (-1 * pow(wn[i][j] - wx[i], 2)) / (2 * pow(bw, 2) ) ) / pow(2 * M_PI * pow(bw, 2), 0.5);
        }
    }

    return;
}   

void rk4_run(int ei, int shell_sample, int band_sample, int neqn, int nval, int nt, double tstart, double dt,
vector<double> tmax, vector<vector<vector<double> > > field_strength, vector<vector<double> > gw, vector<vector<double>>
wn, vector<double> var, vector<vector<double> > wx, vector<arma::mat> Matrix, vector<vector<double> > polarization,
vector<vector<vector<double> > > decay_widths, bool RWA, bool ECALC, bool DECAY, bool TWOPULSE, bool GAUSS, bool
BANDW_AVG, bool STARK, bool WRITE_PULSE, vector<double>& tf_vec, vector<vec1x >& pt_vec_avg, vector<double>& norm_t_vec_avg)
{

    vector<double>          gw_;
    vector<double>          wn_;
    vector<vector<double> > field_strength_;
    vector<double>          wx_;

    if(!TWOPULSE) {
        field_strength_ = vector<vector<double> >(1);
        wx_             = vector<double>(1);
        if(ECALC) {
            cout << "Begining TDSE RK4 dynamics for energy calculation " << ei + 1 << "\n" << endl;
            field_strength_[0] = field_strength[0][0]; 
            gw_                = gw[ei];
            if(BANDW_AVG) {
                wn_                = wn[ei];
            }     
            wx_[0]             = wx[0][ei];
        }
        if(!ECALC) {
            cout << "Begining TDSE RK4 dynamics for intensity calculation " << ei + 1 << "\n" << endl;
            field_strength_[0]  = field_strength[0][ei];
            gw_                 = gw[0];
            if(BANDW_AVG) {
                wn_                 = wn[0];
            }      
            wx_[0]              = wx[0][0];
        }
    }
    if(TWOPULSE) {
        field_strength_ = vector<vector<double> >(2);
        wx_             = vector<double>(2);
        gw_                = gw[0];
        field_strength_[0] = field_strength[0][0];
        field_strength_[1] = field_strength[1][0];
        wx_[0]             = wx[0][0];
        wx_[1]             = wx[1][0];
    }
    vector<vector<vector<vec1x > > > pt_vec(shell_sample, vector<vector<vec1x > > (band_sample, vector<vec1x > (neqn, vec1x (nt, complexd(0.0,0.0)))));

    vector<vector<vector<double> > > norm_t_vec(shell_sample, vector<vector<double> >(band_sample, vector<double>(nt, 0.0)));
    
    vector<double> norm_t  (nt, 0.0);
  	const complex<double> I(0,1);
    vector<vector<double> > field (wx_.size(), vector<double> (nt, 0.0));

    #pragma omp parallel for collapse(2)
    for(int a = 0; a < shell_sample; a++) {
        for(int b = 0; b < band_sample; b++) {

            vector<vec1x > pt(neqn, vec1x (nt, complexd(0.0,0.0)));	
		    vec1x y (neqn, complexd(0.0,0.0));	
    	    y[0] = 1.0; //initial condition - all in the G-state

		    for(int i = 0; i<nt; i++) { 

		        double t0 = tstart + i*dt;
			    double tf = t0 + dt;
                
			    vector<double> Et;
                if(!TWOPULSE) {
                    Et = vector<double>(1);
                    if (GAUSS) {
                        Et[0] = gaussian_field(tf, tmax[0], field_strength_[0][a] * gw_[b], var[0]);
                    }
                    if (!GAUSS) {
                        Et[0] = field_strength_[0][a] * gw_[b];
                    }      
                }
                if(TWOPULSE) {
                    Et = vector<double>(2);
                    if (GAUSS) {
                        Et[0] = gaussian_field(tf, tmax[0], field_strength_[0][a] * gw_[b], var[0]);
                        Et[1] = gaussian_field(tf, tmax[1], field_strength_[1][a] * gw_[b], var[1]);
                    }
                    if (!GAUSS) {
                        Et[0] = field_strength_[0][a] * gw_[b];
                        Et[1] = field_strength_[1][a] * gw_[b];
                    }
                }
                if(BANDW_AVG) {
                    wx_[0] = wn_[b];
                }

			    rk4(neqn, nval, y,  t0, tf, Matrix, polarization, Et, wx_, decay_widths, RWA, DECAY, TWOPULSE, STARK);
               
                if(ei == 0 and a == 0 and b == 0) {
                    for(int n = 0; n < static_cast<int>(field.size()); n++) {
                        if(!RWA) {
                            field[n][i] = Et[n] * cos(wx_[n] * tf);

                            }           
                        }
                }
                     
		        tf_vec[i] = tf * 0.0241;

			    norm_t[i]  = 0.0;
		
    		    for (int j = 0; j<neqn; j++) {                        
				    pt[j][i]    = std::norm(y[j]);
				    norm_t[i]  += pt[j][i].real();

			    }
  
    		    for (int j = 0; j<neqn; j++) pt_vec[a][b][j][i] = pt[j][i];
			    norm_t_vec[a][b][i] = norm_t[i];
                        
		    }
        }
    }
          
    //sum the intensiites for avergaing over the focal volume    
    cout << "Sum populations from focal-voulme/bandwidth averaging" << endl;
    for(int i = 0; i<nt; i++) {
        for (int j = 0; j<neqn; j++) {
            for (int a = 0; a < shell_sample; a++) {
                for (int b = 0; b < band_sample; b++) {
                    pt_vec_avg[j][i]  += pt_vec[a][b][j][i];
                }      
            }
            pt_vec_avg[j][i] /= (shell_sample * band_sample);

        }
    }
    for(int i = 0; i<nt; i++) {
        for (int a = 0; a < shell_sample; a++) {
            for (int b = 0; b < band_sample; b++) {
                norm_t_vec_avg[i] += norm_t_vec[a][b][i];
            }        
        }

        norm_t_vec_avg[i] /= (shell_sample * band_sample);
    }
    if(WRITE_PULSE) {
        if(ei == 0) {
            for(int n = 0; n < static_cast<int>(field.size()); n++) {
                write_field("outputs/gnu/pulse_"+convertInt(n)+".txt", nt, 100, tf_vec, field[n]);
            }
        }
    }
    cout << "Sum complete\n" << endl;

    return;

}



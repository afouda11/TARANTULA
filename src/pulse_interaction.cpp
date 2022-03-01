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


TDSEUTILITY::TDSEUTILITY(void){

}

double TDSEUTILITY::icalib(vector<arma::mat> Matrix, vector<vector<double> > mu, std::vector<vector<double> > wx, double spot_size, vector<double> var) 
{

	int col1, col2;
	std::vector<int>    one;
	std::vector<int>    two;
	std::ifstream in("inputs/icalib_states.txt");
	while(!in.eof()){
		in >> col1;
		one.push_back(col1);
		in >> col2;
		two.push_back(col2);
	}
	double tdm_1, tdm_2;
	if(mu[0][0] == 1.0) {
		tdm_1 = Matrix[0](one[0], two[0]);
		tdm_2 = Matrix[0](one[1], two[1]);
	}
	if(mu[0][1] == 1.0) {
		tdm_1 = Matrix[1](one[0], two[0]);
		tdm_2 = Matrix[1](one[1], two[1]);
	}
	if(mu[0][2] == 1.0) {
		tdm_1 = Matrix[2](one[0], two[0]);
		tdm_2 = Matrix[2](one[1], two[1]);
	}
	double omega = wx[0][0];
	
	double sigma_1 = ( (4 * M_PI * M_PI ) / (3 * 137 ) ) * omega * abs(pow(tdm_1, 2));
	double sigma_2 = ( (4 * M_PI * M_PI ) / (3 * 137 ) ) * omega * abs(pow(tdm_2, 2));

	double fluence_both = sqrt(1 /(sigma_1 * sigma_2));
    spot_size  = (spot_size * 18897.161646321) / (2 * pow(2 * log(2), 0.5));
    //spot_size  = pow(spot_size,2);
	double pulse_energy	= fluence_both * spot_size;
	double peak_power = pulse_energy / var[0];
	double intensity = peak_power / spot_size;

	cout << "Calibrated pulse intensity (W/cm^2)" << endl;
	cout << intensity * 3.50944758E+16 << endl;

	return intensity * 3.50944758E+16 ;

}

void TDSEUTILITY::focal_volume_average(double spot_size, vector<vector<double> >& field_strength, vector<double> intensity, int shell_sample)
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
    return;}

void TDSEUTILITY::bandwidth_average(double bw, std::vector<vector<double> >& gw, std::vector<vector<double> >& wn, std::vector<double> wx, vector<double> bandwidth_avg)
{
    cout << "Sampling " << wx.size() << " central photon energies." << endl;
    cout << "for each central photon energy the " << bw * 27.2114 << " eV " << "bandwith effect will sample " << bandwidth_avg[0] << " energies.\n" << endl;
    for(int i = 0; i < static_cast<int>(wx.size()); i++) {
        double step = ((wx[i] + (bandwidth_avg[1] * bw)) - (wx[i] - (bandwidth_avg[1] * bw))) / bandwidth_avg[0];
        for(int j = 0; j < static_cast<int>(gw[i].size()); j++) {
             wn[i][j] = (wx[i] - (3 * bw)) + (j * step);
             gw[i][j] = exp( (-1 * pow(wn[i][j] - wx[i], 2)) / (2 * pow(bw, 2) ) ) / pow(2 * M_PI * pow(bw, 2), 0.5);
        }
    }

    return;
}   

void TDSEUTILITY::eom_run(int ei, vector<double>& tf_vec, vector<vec1x >& pt_vec_avg, vector<double>& norm_t_vec_avg)
{

    vector<double>          gw_;
    vector<double>          wn_;
    vector<vector<double> > field_strength_;
    vector<double>          wx_;
    
    if(!BOOL_VEC[5]) {
        field_strength_ = vector<vector<double> >(1);
        wx_             = vector<double>(1);
        if(BOOL_VEC[15]) {
            cout << "Begining TDSE RK4 dynamics for energy calculation " << ei + 1 << "\n" << endl;
            field_strength_[0] = field_strength[0][0]; 
            gw_                = gw[ei];
            if(BOOL_VEC[4]) {
                wn_                = wn[ei];
            }     
            wx_[0]             = wx[0][ei];
        }
        if(!BOOL_VEC[15]) {
            cout << "Begining TDSE RK4 dynamics for intensity calculation " << ei + 1 << "\n" << endl;
            field_strength_[0]  = field_strength[0][ei];
            gw_                 = gw[0];
            if(BOOL_VEC[4]) {
                wn_                 = wn[0];
            }      
            wx_[0]              = wx[0][0];
        }
    }
    if(BOOL_VEC[5]) {
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

	int n_decay_chan = static_cast<int>(decay_channels.size());
	int n = neqn / (n_decay_chan+1);
	//const complex<double> I(0,1);
	vector<vector<double> > auger_gamma;
	vector<vector<double> > photo_sigma;
	//vector<vector<double> > photo_sigma;
	vector<vector<vector<double> > > photo_gamma;
	if (BOOL_VEC[2]) {
		if(!BOOL_VEC[5]) {
			auger_gamma = vector<vector<double> > (1, vector<double>(n, 0.0));
			photo_sigma = vector<vector<double> > (1, vector<double>(n, 0.0));
			//photo_gamma = vector<vector<double> > (1, vector<double>(n, 0.0));
			photo_gamma = vector<vector<vector<double> > > (1, vector<vector<double> >(n, vector<double>(nt, 0.0)));
			for(int i = 0; i < n; i++) {
				//auger_gamma[0][i] = decay_widths[0][0][i] / 27.2114;
				auger_gamma[0][i] = decay_widths[0][0][i];
				photo_sigma[0][i] = decay_widths[0][1][i] / 28.0175; //convert megabarn to a.u.
			}
		}
		if (BOOL_VEC[5]) {
			auger_gamma = vector<vector<double> > (2, vector<double>(n, 0.0));
			photo_sigma = vector<vector<double> > (2, vector<double>(n, 0.0));
			//photo_gamma = vector<vector<double> > (2, vector<double>(n, 0.0));
			photo_gamma = vector<vector<vector<double> > > (2, vector<vector<double> >(n, vector<double>(nt, 0.0)));
			for(int i = 0; i < n; i++) {
				auger_gamma[0][i] = decay_widths[0][0][i] / 27.2114;
				auger_gamma[1][i] = decay_widths[1][0][i] / 27.2114;
				photo_sigma[0][i] = decay_widths[0][1][i] / 28.0175;
				photo_sigma[1][i] = decay_widths[1][1][i] / 28.0175;
			}
		}
	}
	if (!BOOL_VEC[2]) {
		if(!BOOL_VEC[5]) {
			auger_gamma = vector<vector<double> > (1, vector<double>(n, 0.0));
			photo_sigma = vector<vector<double> > (1, vector<double>(n, 0.0));
		}
		if(BOOL_VEC[5]) {
			auger_gamma = vector<vector<double> > (2, vector<double>(n, 0.0));
			photo_sigma = vector<vector<double> > (2, vector<double>(n, 0.0));
		}
	}

	EOMDRIVER DRIVEEOM;
	DRIVEEOM.n = n;
	DRIVEEOM.Matrix = Matrix;
	DRIVEEOM.mu = mu;
	DRIVEEOM.auger_gamma = auger_gamma;
	if (!BOOL_VEC[5]) {
		DRIVEEOM.photo_gamma = vector<vector<double> > (1, vector<double>(n, 0.0));
	}
	if (BOOL_VEC[5]) {
		DRIVEEOM.photo_gamma = vector<vector<double> > (2, vector<double>(n, 0.0));
	}
	DRIVEEOM.decay_channels = decay_channels;
	DRIVEEOM.BOOL_VEC = BOOL_VEC;

    #pragma omp parallel for collapse(2)
    for(int a = 0; a < shell_sample; a++) {
        for(int b = 0; b < band_sample; b++) {

            vector<vec1x > pt(neqn, vec1x (nt, complexd(0.0,0.0)));	
		    vec1x y (neqn, complexd(0.0,0.0));	
    	    y[0] = 1.0; //initial condition - all in the G-state
			//y[9] = 1.0;
		    for(int i = 0; i<nt; i++) { 

		        double t0 = tstart + i*dt;
			    double tf = t0 + dt;
                
			    vector<double> Et;
                if (!BOOL_VEC[5]) {//ONE PULSE
                    Et = vector<double>(1);
                    if (BOOL_VEC[1]) {
                        Et[0] = gaussian_field(tf, tmax[0], field_strength_[0][a] * gw_[b], var[0]);
                    }
                    if (!BOOL_VEC[1]) {
                        Et[0] = field_strength_[0][a] * gw_[b];
                    }      
                }
                if (BOOL_VEC[5]) {//TWO PULSE
                    Et = vector<double>(2);
                    if (BOOL_VEC[1]) {//GAUSS
                        Et[0] = gaussian_field(tf, tmax[0], field_strength_[0][a] * gw_[b], var[0]);
                        Et[1] = gaussian_field(tf, tmax[1], field_strength_[1][a] * gw_[b], var[1]);
                    }
                    if (!BOOL_VEC[1]) {//NO GAUSS 
                        Et[0] = field_strength_[0][a] * gw_[b];
                        Et[1] = field_strength_[1][a] * gw_[b];
                    }
                }
                if (BOOL_VEC[4]) {//BANDWIDTH AVERAGE
                    wx_[0] = wn_[b];
                }
				DRIVEEOM.Et = Et;
				DRIVEEOM.wx = wx_;

				if (BOOL_VEC[2]) {//DECAY WIDTHS
					if(!BOOL_VEC[5]) {//ONE PULSE
						for(int states = 0; states < n; states++) {
							photo_gamma[0][states][i] = (photo_sigma[0][states] / wx_[0]) * pow(Et[0],2);
						}
					}
					if (BOOL_VEC[5]) {//TWO PULSE
						for(int states = 0; states < n; states++) {
							photo_gamma[0][states][i] = (photo_sigma[0][states] / wx_[0]) * pow(Et[0],2);
							photo_gamma[1][states][i] = (photo_sigma[1][states] / wx_[1]) * pow(Et[1],2);
						}
					}
				}
	/*			if (!BOOL_VEC[2]) {//NO DECAY WIDTHS
					if(!BOOL_VEC[5]) {//ONE PULSE
						photo_gamma = vector<vector<double> > (1, vector<double>(n, 0.0));
					}
					if(BOOL_VEC[5]) {//TWO PULSE
						photo_gamma = vector<vector<double> > (2, vector<double>(n, 0.0));
					}
				}*/
				for(int pulses = 0; pulses < static_cast<int>(field.size()); pulses++) {
					for(int states = 0; states < n; states++) {
						DRIVEEOM.photo_gamma[pulses][states] = photo_gamma[pulses][states][i];
              		}
				}
				DRIVEEOM.photo_gamma_vec = photo_gamma;
				DRIVEEOM.RK4(y, t0, tf);

                if(ei == 0 and a == 0 and b == 0) {
                    for(int pulses = 0; pulses < static_cast<int>(field.size()); pulses++) {
                        if(!BOOL_VEC[0]) {//NO RWA
                            field[pulses][i] = Et[pulses] * cos(wx_[pulses] * tf);
						}           
					}
                }
                     
		        //tf_vec[i] = tf * 0.0241;
		        tf_vec[i] = tf;

			    norm_t[i]  = 0.0;
				/*	
    		    for (int j = 0; j<neqn; j++) {                        
				    pt[j][i]    = std::norm(y[j]);
				    norm_t[i]  += pt[j][i].real();

			    }*/
    		    for (int j = 0; j<n; j++) {                        
				    pt[j][i]    = std::norm(y[j]);
			    }
				if (BOOL_VEC[13]) {//POPULATION LOSS CHANNELS
					for(int k = 0; k < n_decay_chan; k++) {
						for (int j = (n * (k+1)); j < n * (k+2); j++) {

							//pt[j][i] = DRIVEEOM.Analytical_Population_Loss(tf, j, k);
							pt[j][i] = DRIVEEOM.Numerical_Population_Loss(i, j, k, tf_vec[i]-tf_vec[i-1], pt[j-(n*(k+1))]);

						}
					}
				}
    		    for (int j = 0; j<neqn; j++) {                        
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
    if (BOOL_VEC[11]) {//WRITW FIELD
        if(ei == 0) {
            for(int n = 0; n < static_cast<int>(field.size()); n++) {
                write_field("outputs/gnu/pulse_"+convertInt(n)+".txt", nt, 100, tf_vec, field[n]);
            }
        }
    }
    cout << "Sum complete\n" << endl;

    return;

}

double TDSEUTILITY::gaussian_field(double t, double t_max, double amp_max, double var)
{
   	double Et = amp_max * exp(-1 * (pow((t - t_max),2)) / (2 * (pow(var,2))));

	return Et;
}

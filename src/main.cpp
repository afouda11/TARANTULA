#define _USE_MATH_DEFINES
#include "read_and_write.h"
#include "pulse_interaction.h"
#include "rk4.hpp"
#include "vectypedef.hpp"

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
    cout << "\n\n*!*!*!*!*!*!*!* TDSE(RK4) Solver for XFEL experiments *!*!*!*!*!*!*!*\n\n" << endl;

	//Read in bool options
    vector<bool> BOOL_VEC(16);
    //vector<bool> * BOOL_VEC = new vector<bool>(16);
    BOOL_VEC[0]  = read_bool_options("RWA");
    BOOL_VEC[1]  = read_bool_options("GAUSS");
    BOOL_VEC[2]  = read_bool_options("DECAY");
    BOOL_VEC[3]  = read_bool_options("FOCAL_AVG");
    BOOL_VEC[4]  = read_bool_options("BANDW_AVG");
    BOOL_VEC[5]  = read_bool_options("TWOPULSE");
    BOOL_VEC[6]  = read_bool_options("PERP_AVG");
    BOOL_VEC[7]  = read_bool_options("SUM");
    BOOL_VEC[8]  = read_bool_options("PRINT_MAT");
    BOOL_VEC[9]  = read_bool_options("TWOSTATE");
    BOOL_VEC[10] = read_bool_options("STARK");
    BOOL_VEC[11] = read_bool_options("WRITE_PULSE");
    BOOL_VEC[12] = read_bool_options("PAIR_SUM");
    BOOL_VEC[13] = read_bool_options("DECAY_AMP");
	BOOL_VEC[14] = read_bool_options("ICALIB");

	//Force correct two-state simulation parameters
    if (BOOL_VEC[9]) {//TWO STATE
		BOOL_VEC[6]  = false;//NO PERPENDICULAR AVERAGE
        BOOL_VEC[7]  = false;//NO GROUP SUM
        BOOL_VEC[12] = false;//NO PAIR SUM
		BOOL_VEC[14] = false;//NO INTENSITY CALIBRATION
    }

	//Read and process pulse information
    vector<double> fwhm;
    vector<double> var;
    vector<double> tmax;
    vector<double> width;
    vector<double> spot;
	int n_pulse;
    if (!BOOL_VEC[5]) {//ONE PULSE
        cout << "1 Pulse caclulcation.\n" << endl;
        vector<double> pulse1;
        file2vector("inputs/pulse_1.txt", pulse1);
        fwhm  = vector<double>(1);
        var   = vector<double>(1);
        tmax  = vector<double>(1);
        width = vector<double>(1);
        spot  = vector<double>(1);
        fwhm[0]  = pulse1[0];
        tmax[0]  = pulse1[1];
		width[0] = pulse1[2];
		spot[0]  = pulse1[3];
		n_pulse = 1;
    }
    if (BOOL_VEC[5]) {//TWO PULSE
        cout << "2 Pulse caclulcation. Currently only one intensity and energy can be used for each pulse," << endl;
        cout << "and no bandwidth or focal volume averaging effects can be applied at this current time.\n" << endl;
        vector<double> pulse1;
        vector<double> pulse2;
        file2vector("inputs/pulse_1.txt", pulse1);
        file2vector("inputs/pulse_2.txt", pulse2);
        fwhm  = vector<double>(2);
        var   = vector<double>(2);
        tmax  = vector<double>(2);
        width = vector<double>(2);
        spot  = vector<double>(2);
        fwhm[0]  = pulse1[0]; 
        fwhm[1]  = pulse2[0]; 
        tmax[0]  = pulse1[1]; 
        tmax[1]  = pulse2[1]; 
        width[0] = pulse1[2]; 
        width[1] = pulse2[2]; 
        spot[0]  = pulse1[3]; 
        spot[1]  = pulse2[3];
		n_pulse = 2;
    }
    for (int i = 0; i < static_cast<int>(var.size()); i++) {
        var[i]  = (fwhm[i] * 41.34137) / (2 * pow(2 * log(2), 0.5));  //convert FWHM to sigma
        tmax[i] *= 41.34137;
    }

	//Read and process time information
    vector<double> time_info;
    double tstart;
    file2vector("inputs/time_info.txt", time_info);
    if (time_info[0] == 0.0) {
		if (BOOL_VEC[1]) {//GAUSSIAN PULSE
        	tstart = tmax[0] - (6 * var[0]); //by default pulse 1 arrives before pulse 2
		}
		if (!BOOL_VEC[1]) {//CONSTANT FIELD
        	tstart = 0.0;
		}
    }
    else {
        tstart = time_info[0] * 41.34137;
    }
    double tend = time_info[1] * 41.34137;
    double dt   = time_info[2] * 41.34137;
    int nt      = round((tend-tstart)/dt);
    int n_print = time_info[3]; //the every nth timestep that get printed

	//Read in number of states, pairs and groups
    std::vector<int> n_states;       
    file2vector("inputs/n_states.txt", n_states);
    int neqn       = n_states[0];
    int n_sum_type = n_states[1];
    int n_type     = n_states[2];
	if (!BOOL_VEC[9]) {
    	cout << "The system involves " << neqn << " electronic states" << endl;
	}
    if (BOOL_VEC[9]) {
        neqn = 2;
        n_type = 2;
		cout << "Two state system simulation" << endl;
    }

	//Read x, y and z direction matrices: see read_matrix.cpp   
    vector<arma::mat> Matrix (3); 
	if (!BOOL_VEC[9]) {//NO TWO STATE
		Matrix[0]  = read_matrix(neqn, "matrix_elements/Diagonal.txt", "matrix_elements/Off_Diagonal_x.txt");
		Matrix[1]  = read_matrix(neqn, "matrix_elements/Diagonal.txt", "matrix_elements/Off_Diagonal_y.txt");
		Matrix[2]  = read_matrix(neqn, "matrix_elements/Diagonal.txt", "matrix_elements/Off_Diagonal_z.txt");
	}
	if (BOOL_VEC[9]) {//TWO STATE
		Matrix[0]  = read_matrix(neqn, "matrix_elements/Diagonal_2STATE.txt", "matrix_elements/Off_Diagonal_x_2STATE.txt");
		Matrix[1]  = read_matrix(neqn, "matrix_elements/Diagonal_2STATE.txt", "matrix_elements/Off_Diagonal_y_2STATE.txt");
		Matrix[2]  = read_matrix(neqn, "matrix_elements/Diagonal_2STATE.txt", "matrix_elements/Off_Diagonal_z_2STATE.txt");
	}
    if (BOOL_VEC[8]) {//PRINT MATRIX
        cout << Matrix[0];
    }      
    cout << "Matrix elements read\n" << endl;

	//Read light polarization vector
	vector<vector<double> > mu;
    if (!BOOL_VEC[5]) {//ONE PULSE
        mu = vector<vector<double> >(1);
        file2vector("inputs/polarization_1.txt", mu[0]); 
    }
    if (BOOL_VEC[5]) {//TWO PULSE
        mu = vector<vector<double> >(2);
        file2vector("inputs/polarization_1.txt", mu[0]); 
        file2vector("inputs/polarization_2.txt", mu[1]); 
    }
	//TDSE utility object, runs EOM derivatives etc.
	TDSEUTILITY UTILITYTDSE;

    //Central photon energy and bandwidth sampling
    std::vector<vector<double> > gw; //for bandwidth effect and 2 pulse this would need to be a vecvecvec(double)
    std::vector<vector<double> > wn; //for bandwidth effect and 2 pulse this would need to be a vecvecvec(double)
    std::vector<vector<double> > wx;     
    int n_photon_e  = 1;
    int band_sample = 1;
    if (!BOOL_VEC[5]) {//ONE PULSE
        wx = std::vector<vector<double> >(1);     
        file2vector("inputs/photon_e_1.txt", wx[0]);
        n_photon_e = wx[0].size();
		cout << "central photon energies (eV):" << endl;
	    for(int i = 0; i < n_photon_e; i++) {
			cout << wx[0][i] << endl;
			wx[0][i] /=  27.2114;
		}	
		cout << "\n";
        if (BOOL_VEC[4]) {//BANDWIDTH AVERAGE
        	vector<double> bandwidth_avg;
        	file2vector("inputs/bandwidth_avg.txt", bandwidth_avg);
            band_sample = bandwidth_avg[0];
            double bw = (width[0] / 27.2114) / (2 * pow(2 * log(2), 0.5));
            gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  0.0));
            wn = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  0.0));
            UTILITYTDSE.bandwidth_average(bw, gw, wn, wx[0], bandwidth_avg);
        }
        if (!BOOL_VEC[4]) {//NO BANDWIDTH AVERAGE
            cout << "No Bandwidth effect applied\n" << endl;
            band_sample = 1;
            gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  1.0));
        }
    }
    if (BOOL_VEC[5]) {//TWO PULSE
        wx = std::vector<vector<double> >(2);     
        file2vector("inputs/photon_e_1.txt", wx[0]);
        file2vector("inputs/photon_e_2.txt", wx[1]);
		n_photon_e = 1;
		cout << "central photon energy for pulse 1 (eV):" << endl;
		cout << wx[0][0] << endl;
		wx[0][0] /=  27.2114;
		cout << "central photon energy for pulse 2 (eV):" << endl;
		cout << wx[1][0] << endl;
		wx[1][0] /=  27.2114;
        cout << "No Bandwidth effect applied\n" << endl;
        band_sample = 1;
        gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  1.0));
    }

    //Peak intensity and focal volume averaging
    std::vector<vector<double> > intensity;          //intensity
    vector<vector<vector<double> > > field_strength;	
    int n_intensity  = 1;
    int shell_sample = 1;
    double spot_size = spot[0];                 //micron
    if (!BOOL_VEC[5]) {//ONE PULSE
        intensity = std::vector<vector<double> >(1);     
        file2vector("inputs/intensity_1.txt", intensity[0]);
        n_intensity = intensity[0].size();
		cout << "Pulse intensities from input file (W/cm^2):" << endl;
	    for(int i = 0; i < n_intensity; i++) {
			cout << intensity[0][i] << endl;
		}
		if (BOOL_VEC[14]) {//INTENSITY CALIBRATION
		cout << "Intensity calibrated from saturation fluence of selected transitions" << endl;
			intensity[0][0] = UTILITYTDSE.icalib(Matrix, mu, wx, spot_size, var);
		}

        if (BOOL_VEC[3]) {
        	vector<int> focalvol_avg;
        	file2vector("inputs/focalvol_avg.txt", focalvol_avg);
            shell_sample = focalvol_avg[0];
            field_strength = vector<vector<vector<double> > > (1,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
            UTILITYTDSE.focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample);
        }
        if (!BOOL_VEC[3]) {
            cout << "No focal volume effect applied\n" << endl;
            shell_sample = 1;
            field_strength = vector<vector<vector<double> > > (1,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
            UTILITYTDSE.focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample);
        }
    }
    if (BOOL_VEC[5]) {
        intensity = std::vector<vector<double> >(2);     
        file2vector("inputs/intensity_1.txt", intensity[0]);
        file2vector("inputs/intensity_2.txt", intensity[1]);
        n_intensity = 1;
		cout << "Intensity for pulse 1 (W/cm^2):" << endl;
		cout << intensity[0][0] << endl;
		cout << "Intensity for pulse 2 (W/cm^2):" << endl;
		cout << intensity[1][0] << endl;
        cout << "No focal volume effect applied\n" << endl;
        shell_sample = 1;
        field_strength = vector<vector<vector<double> > > (2,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
        UTILITYTDSE.focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample); 
        UTILITYTDSE.focal_volume_average(spot_size, field_strength[1], intensity[1], shell_sample); 
    }

    //Auger decay widths and phtotoionisation cross-sections
    std::vector<vector<vector<double> > > decay_widths; 
    if (!BOOL_VEC[5]) {
        decay_widths = vector<vector<vector<double> > > (1, vector<vector<double> >(2));
        file2vector("inputs/auger_rates_1.txt", decay_widths[0][0]);
        file2vector("inputs/photoion_sigma_1.txt", decay_widths[0][1]);
    }
    if (BOOL_VEC[5]) {
        decay_widths = vector<vector<vector<double> > > (2, vector<vector<double> >(2));
        file2vector("inputs/auger_rates_1.txt", decay_widths[0][0]);
        file2vector("inputs/auger_rates_2.txt", decay_widths[1][0]);
        file2vector("inputs/photoion_sigma_1.txt", decay_widths[0][1]);
        file2vector("inputs/photoion_sigma_2.txt", decay_widths[1][1]);
    }
    cout << "Decay widths read\n" << endl;

    //Calculate and print population loss channels
    std::vector<string> decay_channels;
    int n_decay_chan = 0;
    if (BOOL_VEC[13]) {
        file2vector("inputs/decay_channels.txt", decay_channels);
        n_decay_chan = static_cast<int>(decay_channels.size());
        cout << "Decay amplitude channles included in simulation:\n" << endl;
        for(int i = 0; i < n_decay_chan; i++) {
            cout << decay_channels[i] << endl;
        } 
         cout << "\n";
		
        neqn       += neqn       * n_decay_chan;
        n_type     += n_type     * n_decay_chan;
        n_sum_type += n_sum_type * n_decay_chan;
    }
	
	//Sample single or multiple photon energies and or intensities
    int  n_calc = 0;
    //bool ECALC  = true;
	BOOL_VEC[15] = true;
    if ( n_intensity > 1 and n_photon_e == 1 ) {
        BOOL_VEC[15] = false;
        n_calc = n_intensity;
        cout << "Simulation across " << n_calc <<" intensities\n" << endl;
    }
    else if ( n_photon_e >= 1 and n_intensity == 1 ) {
            if ( n_photon_e == 1 and n_intensity == 1 ) {
                cout << "Single calculation (1 photon energy and intensity)\n" << endl; 
                BOOL_VEC[15] = true;   //default 
                n_calc = n_photon_e;        
            }
            else if (n_photon_e > 1) {
                BOOL_VEC[15] = true;    
                n_calc = n_photon_e; 
                cout << "Simulation across " << n_calc <<" central photon energies\n" << endl;
            }
    }

	//Print time information
    cout << "Time details:" << endl;
    cout << "Start time: "  << tstart << " a.u " << " / " << tstart/41.34137 << " fs"<< endl;
    cout << "End time: "    << tend   << " a.u " << " / " << tend/41.34137   << " fs"<< endl;
    cout << "Time Step: "   << dt     << " a.u " << " / " << dt/41.34137     << " fs\n"<< endl;
    cout << "Number of steps:" << nt << endl;
    cout << "Print every " << n_print << " steps\n" << endl;

	//Print pulse information
    cout << "Pulse details:" << endl;
    for (int i = 0; i < static_cast<int>(var.size()); i++) {
        cout << "*** Pulse " <<  i + 1 << " ***" <<endl;
        cout << "FWHM: "  << fwhm[i] * 41.34137 << " a.u " << " / " << fwhm[i] << " fs " << endl;
        cout << "sigma: " << var[i]  << " a.u " << " / " << var[i] * 0.02418884254 << " fs " << endl;
        cout << "Tmax: "  << tmax[i] << " a.u " << " / " << tmax[i] * 0.02418884254 << " fs\n" << endl;
    }

    //TDSE vector containers
    vector<double> tf_vec(nt, 0.0);
    //involves the summed degenerate paris of final states
    vector<vector<vec1x > > pt_vec(n_calc, vector<vec1x > (n_type, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<vec1x > > pt_vec_avg(n_calc, vector<vec1x > (neqn, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<double> > norm_t_vec_avg(n_calc, vector<double> (nt, 0.0));
    //used only for averging over x and y initated by PEP_AVG
    vector<vector<vec1x > > pt_vec_avg_perp(n_calc, vector<vec1x > (neqn, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<vec1x > > pt_vec_perp(n_calc, vector<vec1x > (n_type, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<double> > norm_t_vec_avg_perp(n_calc, vector<double> (nt, 0.0));
    //for group_sum, if SUM=true
    vector<vector<vec1x > > pt_sum_vec(n_calc, vector<vec1x > (n_sum_type, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<vec1x > > pt_sum_vec_perp(n_calc, vector<vec1x > (n_sum_type, vec1x (nt, complexd(0.0,0.0))));

	//vector<bool> *BOOLS;
	//BOOLS = &BOOL_VEC;
	//WRITE SUMMED data to files 
	FILEWRITER WRITEFILES;	
	WRITEFILES.nt = nt;
	WRITEFILES.n_type = n_type;
	WRITEFILES.n_sum_type = n_sum_type;
	WRITEFILES.n_print = n_print;
	WRITEFILES.BOOL_VEC = BOOL_VEC;
	WRITEFILES.n_photon_e = n_photon_e;
	WRITEFILES.n_calc = n_calc;
	WRITEFILES.neqn = neqn;
	if (BOOL_VEC[15]) {
		WRITEFILES.variable = wx[0];
		WRITEFILES.varstring = "energy";
	}
	if (!BOOL_VEC[15]) {
		WRITEFILES.variable = intensity[0];
		WRITEFILES.varstring = "intensity";
	}
	UTILITYTDSE.shell_sample = shell_sample;
	UTILITYTDSE.band_sample  = band_sample;
	UTILITYTDSE.neqn = neqn;
	UTILITYTDSE.nt = nt;
	UTILITYTDSE.tstart = tstart;
	UTILITYTDSE.dt = dt;
	UTILITYTDSE.tmax = tmax;
	UTILITYTDSE.field_strength = field_strength;
	UTILITYTDSE.gw = gw;
	UTILITYTDSE.wn = wn;
	UTILITYTDSE.var = var;
	UTILITYTDSE.wx = wx;
	UTILITYTDSE.Matrix = Matrix;
	UTILITYTDSE.mu = mu;
	UTILITYTDSE.decay_widths = decay_widths;
	UTILITYTDSE.decay_channels = decay_channels;
	UTILITYTDSE.BOOL_VEC = BOOL_VEC;
	UTILITYTDSE.n_pulse = n_pulse;

    clock_t startTime;    
    for(int ei = 0; ei < n_calc; ei++) {

        //DO THE TDSE
        startTime = clock();    

        UTILITYTDSE.eom_run(ei, tf_vec, pt_vec_avg[ei], norm_t_vec_avg[ei]);

        if(BOOL_VEC[6]) {
            mu[0][0] = 0.0; mu[0][1] = 1.0; mu[0][2] = 0.0;
            if(BOOL_VEC[5]) {//not properly implemented for two pulse, assumes both pusles perpendicular to z
                mu[1][0] = 0.0; 
				mu[1][1] = 1.0; 
				mu[1][2] = 0.0;
            }
			UTILITYTDSE.mu = mu;

            UTILITYTDSE.eom_run(ei, tf_vec, pt_vec_avg_perp[ei], norm_t_vec_avg_perp[ei]);    

            //return to x
            mu[0][0] = 1.0; 
			mu[0][1] = 0.0; 
			mu[0][2] = 0.0;
            if(BOOL_VEC[5]) {//not properly implemented for two pulse, assumes both pusles perpendicular to z
            	mu[1][0] = 1.0; 
				mu[1][1] = 0.0; 
				mu[1][2] = 0.0;
            }
        }

        cout << "RK4 Time:\n" << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds.\n" << endl;

        //SUM groups of states listed input/group_names.txt and correpseonding _index.txt files to neaten represenation
        if (BOOL_VEC[7]) {
            group_sum(n_sum_type, nt, neqn, n_decay_chan, pt_sum_vec[ei], pt_vec_avg[ei]);
            if(BOOL_VEC[6]) {
                group_sum(n_sum_type, nt, neqn, n_decay_chan, pt_sum_vec_perp[ei], pt_vec_avg_perp[ei]);
            }
            cout << "Sum complete" << endl;
        }
        //sum degenerate pairs of pi states, need to list the pairs in inputs/state_pairs.txt
        if (!BOOL_VEC[9]) {
            if (BOOL_VEC[12]) {
                pair_sum(nt, n_type, n_decay_chan, pt_vec[ei], pt_vec_avg[ei]);
            }
            if (!BOOL_VEC[12]) {
                pt_vec = pt_vec_avg;
            }       
        }
        if (BOOL_VEC[9]) {
            pt_vec = pt_vec_avg;
        }
        if (BOOL_VEC[6]) { //twostate and perp_avg are not compatible together
            if (BOOL_VEC[12]) {
                pair_sum(nt, n_type, n_decay_chan, pt_vec_perp[ei], pt_vec_avg_perp[ei]);
            }      
            if (!BOOL_VEC[12]) {
                pt_vec_perp = pt_vec_avg_perp;
            }      
        }       

		WRITEFILES.tf_vec = tf_vec;
		WRITEFILES.write_data_files(convertInt(ei), pt_vec[ei], pt_sum_vec[ei], pt_vec_perp[ei], pt_sum_vec_perp[ei], norm_t_vec_avg[ei], norm_t_vec_avg_perp[ei]);
    }

    if (n_calc > 1) { //only for 1 pulse calc over mutiple intensities or energies
        WRITEFILES.write_data_variable_files(pt_vec, pt_sum_vec, pt_vec_perp, pt_sum_vec_perp);
        
    }

    cout << "Simulation complete! Nice warn.\n" << endl;
}


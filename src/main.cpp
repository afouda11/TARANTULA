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

#ifdef _OPENMP
#include <omp.h>
#endif

int main()
{
    cout << "\n\n*!*!*!*!*!*!*!* TDSE(RK4) Solver for XFEL experiments *!*!*!*!*!*!*!*\n\n" << endl;

    bool RWA         = read_bool_options("RWA");      //do rotating wave approx.
    bool GAUSS       = read_bool_options("GAUSS");   
    bool DECAY       = read_bool_options("DECAY");   
    bool FOCAL_AVG   = read_bool_options("FOCAL_AVG");//focal averaging over the intensity volume
    bool BANDW_AVG   = read_bool_options("BANDW_AVG"); 
    bool TWOPULSE    = read_bool_options("TWOPULSE");
    bool PERP_AVG    = read_bool_options("PERP_AVG"); //average over the x and y polarization_1.txt must be set to x
    bool SUM         = read_bool_options("SUM");      //sum the G, Fv, Fr, Iv, Ir and Id state types for neater plotting 
    bool PRINT_MAT   = read_bool_options("PRINT_MAT");   
    bool TWOSTATE    = read_bool_options("TWOSTATE"); //two level calculation on resonance with the first core excitation 
    bool STARK       = read_bool_options("STARK");
    bool WRITE_PULSE = read_bool_options("WRITE_PULSE");
    bool PAIR_SUM    = read_bool_options("PAIR_SUM");
    bool DECAY_AMP   = read_bool_options("DECAY_AMP");

    if (TWOSTATE) {                   
        SUM       = false;
        DECAY     = false;
        FOCAL_AVG = false;
        BANDW_AVG = false;
        PERP_AVG  = false;
        TWOPULSE  = false;
        PAIR_SUM  = false;
    }

    vector<double> fwhm;
    vector<double> var;
    vector<double> tmax;
    if(!TWOPULSE) {
        cout << "1 Pulse caclulcation.\n" << endl;
        vector<double> pulse1;
        file2vector("inputs/pulse_1.txt", pulse1);
        fwhm = vector<double>(1, pulse1[0]);
        var  = vector<double>(1, pulse1[0]);
        tmax = vector<double>(1, pulse1[1]);
    }
    if(TWOPULSE) {
        cout << "2 Pulse caclulcation. Currently only one intensity and energy can be used for each pulse," << endl;
        cout << "and no bandwidth or focal volume averaging effects can be applied at this current time.\n" << endl;
        vector<double> pulse1;
        vector<double> pulse2;
        file2vector("inputs/pulse_1.txt", pulse1);
        file2vector("inputs/pulse_2.txt", pulse2);
        fwhm = vector<double>(2);
        var  = vector<double>(2);
        tmax = vector<double>(2);
        fwhm[0] = pulse1[0]; 
        fwhm[1] = pulse2[0]; 
        tmax[0] = pulse1[1]; 
        tmax[1] = pulse2[1]; 
    }
    for (int i = 0; i < static_cast<int>(var.size()); i++) {
        var[i]  = (fwhm[i] * 41.34137) / (2 * pow(2 * log(2), 0.5));  //convert FWHM to sigma
        tmax[i] *= 41.34137;
    }

    vector<double> time_info;
    double tstart;
    file2vector("inputs/time_info.txt", time_info);
    if(time_info[0] == 0.0) {
        tstart = tmax[0] - (6 * var[0]); //by default pulse 1 arrives before pulse 2
    }
    else {
        tstart = time_info[0] * 41.34137;
    }
    double tend = time_info[1] * 41.34137;
    double dt   = time_info[2] * 41.34137;
    int nt      = round((tend-tstart)/dt);
    int n_print = time_info[3]; //the every nth timestep that get printed

    std::vector<int> n_states;       //read total number and number of valence states from file
    file2vector("inputs/n_states.txt", n_states);
    int neqn       = n_states[0];
    int n_sum_type = n_states[1];
    int n_type     = n_states[2];
    cout << "The system involves " << neqn << " electronic states" << endl;

    if (TWOSTATE) {
        neqn = 2;
        n_type = 2;
    }     

    //central photon energy and bandwidth sampling
    std::vector<vector<double> > gw; //for bandwidth effect and 2 pulse this would need to be a vecvecvec(double)
    std::vector<vector<double> > wn; //for bandwidth effect and 2 pulse this would need to be a vecvecvec(double)
    std::vector<vector<double> > wx;     
    int n_photon_e = 1;
    int band_sample;
    if(!TWOPULSE) {
        wx = std::vector<vector<double> >(1);     
        file2vector("inputs/photon_e_1.txt", wx[0]);
        n_photon_e = wx[0].size();
	    for(int i = 0; i < n_photon_e; i++) wx[0][i] /=  27.2114;
        if(BANDW_AVG) { 
            band_sample = 25;
            double bw = (4.0 / 27.2114) / (2 * pow(2 * log(2), 0.5));
            gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  0.0));
            wn = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  0.0));
            bandwidth_average(bw, gw, wn, wx[0], band_sample);
        }
        if(!BANDW_AVG) {
            cout << "No Bandwidth effect applied\n" << endl;
            band_sample = 1;
            gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  1.0));
        }
    }
    if(TWOPULSE) {
        wx = std::vector<vector<double> >(2);     
        file2vector("inputs/photon_e_1.txt", wx[0]);
        file2vector("inputs/photon_e_2.txt", wx[1]);
	    for(int i = 0; i < n_photon_e; i++) {
            wx[0][i] /=  27.2114;
            wx[1][i] /=  27.2114;
        }
        cout << "No Bandwidth effect applied\n" << endl;
        band_sample = 1;
        gw = std::vector<vector<double> >(n_photon_e, vector<double>(band_sample,  1.0));
    }

    //peak intensity and focal volume averaging
    std::vector<vector<double> > intensity;          //intensity
    vector<vector<vector<double> > > field_strength;
    int n_intensity;
    int shell_sample;
    double spot_size = 2.0;                 //micron
    if(!TWOPULSE) {
        intensity = std::vector<vector<double> >(1);     
        file2vector("inputs/intensity_1.txt", intensity[0]);
        n_intensity = intensity[0].size();

        if(FOCAL_AVG) {
            shell_sample = 25;
            field_strength = vector<vector<vector<double> > > (1,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
            focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample);
        }
        if(!FOCAL_AVG) {
            cout << "No focal volume effect applied\n" << endl;
            shell_sample = 1;
            field_strength = vector<vector<vector<double> > > (1,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
            focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample);
        }
    }
    if(TWOPULSE) {
        intensity = std::vector<vector<double> >(2);     
        file2vector("inputs/intensity_1.txt", intensity[0]);
        file2vector("inputs/intensity_2.txt", intensity[1]);
        n_intensity = 1;
        cout << "No focal volume effect applied\n" << endl;
        shell_sample = 1;
        field_strength = vector<vector<vector<double> > > (2,  vector<vector<double> >(n_intensity, vector<double>(shell_sample, 0.0)));
        focal_volume_average(spot_size, field_strength[0], intensity[0], shell_sample); 
        focal_volume_average(spot_size, field_strength[1], intensity[1], shell_sample); 
    }

    //auger decay widths and phtotoionisation cross-sections
    std::vector<vector<vector<double> > > decay_widths; 
    if(!TWOPULSE) {
        decay_widths = vector<vector<vector<double> > > (1, vector<vector<double> >(2));
        file2vector("inputs/auger_rates_1.txt", decay_widths[0][0]);
        file2vector("inputs/photoion_sigma_1.txt", decay_widths[0][1]);
    }
    if(TWOPULSE) {
        decay_widths = vector<vector<vector<double> > > (2, vector<vector<double> >(2));
        file2vector("inputs/auger_rates_1.txt", decay_widths[0][0]);
        file2vector("inputs/auger_rates_2.txt", decay_widths[1][0]);
        file2vector("inputs/photoion_sigma_1.txt", decay_widths[0][1]);
        file2vector("inputs/photoion_sigma_2.txt", decay_widths[1][1]);
    }

    cout << "Decay widths read\n" << endl;
    vector<string> decay_channels;
    int n_decay_chan = 0;
    if(DECAY_AMP) {
        file2vector("inputs/decay_channels.txt", decay_channels);
        n_decay_chan = static_cast<int>(decay_channels.size());
        cout << "Amplitudes for the following decay channles will be calculated\n" << endl;
        for(int i = 0; i < n_decay_chan; i++) {
            cout << decay_channels[i] << endl;
        } 
         cout << "\n";

        neqn       += neqn       * n_decay_chan;
        n_type     += n_type     * n_decay_chan;
        n_sum_type += n_sum_type * n_decay_chan;
    }

    vector<vector<double> > polarization;
    if(!TWOPULSE) {
        polarization = vector<vector<double> >(1);
        file2vector("inputs/polarization_1.txt", polarization[0]); 
    }
    if(TWOPULSE) {
        polarization = vector<vector<double> >(2);
        file2vector("inputs/polarization_1.txt", polarization[0]); 
        file2vector("inputs/polarization_2.txt", polarization[1]); 
    }

    int ndim = 3;
    size_t ndimall = 4;
    std::string xyzall[ndimall] = {"x", "y", "z", "all"};
    vector<arma::mat> Matrix (ndim); //read  x, y and z direction matrices: see read_matrix.cpp   
    for(int i = 0; i < ndim; i++) {
        if (!TWOSTATE) {
            Matrix[i]  = read_matrix(neqn, "matrix_elements/Diagonal.txt", "matrix_elements/Off_Diagonal_"+xyzall[i]+".txt");
        }
        if (TWOSTATE) {
            Matrix[i]  = read_matrix(neqn, "matrix_elements/Diagonal_2STATE.txt", "matrix_elements/Off_Diagonal_"+xyzall[i]+"_2STATE.txt"); 
        }
    }
    if(PRINT_MAT) {
        cout << Matrix[0];
    }      
    cout << "Matrix elements read\n" << endl;

    int  n_calc = 0;
    bool ECALC  = true;
    if ( n_intensity > 1 and n_photon_e == 1 ) {
        ECALC = false;
        n_calc = n_intensity;
        cout << "Calculation as a function of intensity (" << n_calc <<" intensity calculations)\n" << endl;
    }
    else if ( n_photon_e >= 1 and n_intensity == 1 ) {
            if ( n_photon_e == 1 and n_intensity == 1 ) {
                cout << "Single calculation (1 photon energy and intensity)\n" << endl; 
                ECALC = true;   //default 
                n_calc = n_photon_e;        
            }
            else if (n_photon_e > 1) {
                ECALC = true;    
                n_calc = n_photon_e; 
                cout << "Calculation as a function of photon energy (" << n_calc <<" energy calculations)\n" << endl;
            }
    }

    cout << "Time details:" << endl;
    cout << "Start time: "  << tstart << " a.u " << " / " << tstart/41.34137 << " fs"<< endl;
    cout << "End time: "    << tend   << " a.u " << " / " << tend/41.34137   << " fs"<< endl;
    cout << "Time Step: "   << dt     << " a.u " << " / " << dt/41.34137     << " fs\n"<< endl;
    cout << "Number of steps:" << nt << endl;
    cout << "Print every " << n_print << " steps\n" << endl;

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
    //if SUM = true
    vector<vector<vec1x > > pt_sum_vec(n_calc, vector<vec1x > (n_sum_type, vec1x (nt, complexd(0.0,0.0))));
    vector<vector<vec1x > > pt_sum_vec_perp(n_calc, vector<vec1x > (n_sum_type, vec1x (nt, complexd(0.0,0.0))));

    clock_t startTime;    
    for(int ei = 0; ei < n_calc; ei++) {

        //DO THE TDSE
        startTime = clock();    
        rk4_run(ei, shell_sample, band_sample, neqn, nt, tstart, dt, tmax, field_strength, gw, wn, var, wx, Matrix,
        polarization, decay_widths, decay_channels, RWA, ECALC, DECAY, TWOPULSE, GAUSS, BANDW_AVG, STARK, WRITE_PULSE,
        DECAY_AMP, tf_vec, pt_vec_avg[ei], norm_t_vec_avg[ei]);

        if(PERP_AVG) {
            polarization[0][0] = 0.0; polarization[0][1] = 1.0; polarization[0][2] = 0.0;
            if(TWOPULSE) {//not properly implemented for two pulse, assumes both pusles perpendicular to z
                polarization[1][0] = 0.0; polarization[1][1] = 1.0; polarization[1][2] = 0.0;
            }
            rk4_run(ei, shell_sample, band_sample, neqn, nt, tstart, dt, tmax, field_strength, gw, wn, var, wx, Matrix,
            polarization, decay_widths, decay_channels, RWA, ECALC, DECAY, TWOPULSE, GAUSS, BANDW_AVG, STARK,
            WRITE_PULSE, DECAY_AMP, tf_vec, pt_vec_avg_perp[ei], norm_t_vec_avg_perp[ei]);    
            //return to x
            polarization[0][0] = 1.0; polarization[0][1] = 0.0; polarization[0][2] = 0.0;
            if(TWOPULSE) {//not properly implemented for two pulse, assumes both pusles perpendicular to z
                polarization[1][0] = 1.0; polarization[1][1] = 0.0; polarization[1][2] = 0.0;
            }
        }

        cout << "RK4 Time:\n" << double( clock() - startTime ) / (double)CLOCKS_PER_SEC<< " seconds.\n" << endl;

        //SUM groups of states listed input/group_names.txt and correpseonding _index.txt files to neaten represenation
        if (SUM) {
            group_sum(n_sum_type, nt, neqn, n_decay_chan, pt_sum_vec[ei], pt_vec_avg[ei]);
            if(PERP_AVG) {
                group_sum(n_sum_type, nt, neqn, n_decay_chan, pt_sum_vec_perp[ei], pt_vec_avg_perp[ei]);
            }
            cout << "Sum complete" << endl;
        }
        
        //sum degenerate pairs of pi states, need to list the pairs in inputs/state_pairs.txt
        if (!TWOSTATE) {
            if (PAIR_SUM) {
                pair_sum(nt, n_type, pt_vec[ei], pt_vec_avg[ei]);
            }
            else {
                pt_vec = pt_vec_avg;
            }       
        }
        if (TWOSTATE) {
            pt_vec = pt_vec_avg;
        }
        if(PERP_AVG) { //twostate and perp_avg are not compatible together
            if (PAIR_SUM) {
                pair_sum(nt, n_type, pt_vec_perp[ei], pt_vec_avg_perp[ei]);
            }      
            else {
                pt_vec_perp = pt_vec_avg_perp;
            }      
        }       
        //WRITE SUMMED data to files
      
        FILEWRITER WRITEFILES;
        
         WRITEFILES.nt = nt;
         WRITEFILES.n_type = n_type;
         WRITEFILES.n_sum_type = n_sum_type;
         WRITEFILES.n_print = n_print;
         WRITEFILES.tf_vec = tf_vec;

        WRITEFILES.write_data_files(convertInt(ei), pt_vec[ei], pt_sum_vec[ei], pt_vec_perp[ei], pt_sum_vec_perp[ei], norm_t_vec_avg[ei], norm_t_vec_avg_perp[ei], SUM, PERP_AVG);
    }

    FILEWRITER WRITEFILES;
    if (n_calc > 1) { //only for 1 pulse calc over mutiple intensities or energies
        WRITEFILES.write_data_variable_files(n_photon_e, n_calc, intensity[0], wx[0], pt_vec, pt_sum_vec, pt_vec_perp, pt_sum_vec_perp, SUM, ECALC, PERP_AVG);
        
    }

    cout << "Simulation complete! Nice warn.\n" << endl;
}


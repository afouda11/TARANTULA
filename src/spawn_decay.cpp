/*
TARANTULA
Copyright (C) 2022  Adam E. A. Fouda
This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
*/

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

void SPAWN_DECAY() {

cout << "\n\n*** Spawning Decay Dynamics Implenetation on AIMD data ***\n\n" << endl;

	int neqn;
	int natom;
	double tstart;
	double tend;
	double dt;
	double nt;	
	int nt_spawn; //spawn every nt_spawn steps

 	read_options("NEQN",   		neqn);
 	read_options("NATOM",  		natom);
 	read_options("TSTART", 		tstart);
 	read_options("TEND",   		tend);
 	read_options("DT",     		dt);
 	read_options("NT_SPAWN",	nt_spawn);
	//tstart *= 41.34137;
	//tend   *= 41.34137;
	//dt     *= 41.34137;

	//number of decay channels from initial state
	int nchan;
 	read_options("NCHAN", nchan);
	//number of decay steps involved in each channel (each channel has same numebr of steps in this implementation)
	int nstep;
 	read_options("NSTEP", nstep);

	nt = (tend / dt) + 1;

	//container for data input files
	std::vector<std::vector<double> > data      = std::vector<std::vector<double> >(neqn);
	std::vector<std::vector<double> > data_spin = std::vector<std::vector<double> >(neqn);
	//read datat files
	file2vector("inputs/data_files/data_0_0.txt",  data[0]);
	file2vector("inputs/data_files/data_0_0_spin.txt",  data_spin[0]);
	int count; 
	count = 1;
	for (int i = 1; i < nstep+1; i++) {
		for (int j = 1; j < nchan+1; j++) {
		
			file2vector("inputs/data_files/data_"+convertInt(j)+"_"+convertInt(i)+".txt",  data[count]);
			file2vector("inputs/data_files/data_"+convertInt(j)+"_"+convertInt(i)+"_spin.txt",  data_spin[count]);
			count++;
		}
	}
	//containers for separate time, energy bandlenght and charge variabels from data files fo eahc state
	std::vector<double> time;
	std::vector<std::vector<double> > energy  = std::vector<std::vector<double> >(neqn);
	std::vector<std::vector<double> > kenergy  = std::vector<std::vector<double> >(neqn);
	std::vector<std::vector<double> > bondl  = std::vector<std::vector<double> >(neqn);
	std::vector<std::vector<std::vector<double> > > charge;  
	std::vector<std::vector<std::vector<double> > > spin;  
	charge = std::vector<std::vector<std::vector<double>>>(natom, std::vector<std::vector<double>>(neqn));
	spin   = std::vector<std::vector<std::vector<double>>>(natom, std::vector<std::vector<double>>(neqn));
	for (int i = 0; i < nt; i++) {
		cout << i << endl;
		int t_index  = (i*(4+natom));
		int e_index  = (i*(4+natom))+1;
		int ke_index = (i*(4+natom))+2;
		int b_index  = (i*(4+natom))+3;
		//time same for all data files
		time.push_back(data[0][t_index]);

		for (int j = 0; j < neqn; j++) {

			energy[j].push_back(data[j][e_index]);
			kenergy[j].push_back(data[j][ke_index]);
			bondl[j].push_back(data[j][b_index]);

			for (int k = 0; k < natom; k++) {
				charge[k][j].push_back(data[j][(i*(4+natom))+(k+4)]);
				spin[k][j].push_back(data_spin[j][(i*(4+natom))+(k+4)]);

			}	
		}
	}

	//total Auger rate, which governs the population loss of the inital state
	vector<vector<double> > total_auger_gamma = vector<vector<double> >(neqn);
	file2vector("inputs/total_auger_rates.txt", total_auger_gamma[0]);
	for (int n = 0; n < neqn; n ++) {
		total_auger_gamma[0][n] /= 27.2114;
		//total_auger_gamma[0][n] *= 10000;
	}
	//partial Auger rate for eachchannel of each step, 1 input file has all the channels in 1 step, determines the
	//increase in population of the each state the intial state decays too 
	vector<vector<double> > partial_auger_gamma = vector<vector<double> >(nstep);
	for (int i = 0; i < nstep; i++) {
		file2vector("inputs/partial_auger_rates_"+convertInt(i)+".txt", partial_auger_gamma[i]);
		for (int j = 0; j < nchan; j++) {
			partial_auger_gamma[i][j] /= 27.2114;
			//partial_auger_gamma[i][j] *= 10000;

		}
	}

	//vector containing number of trajectory basis functions for each state
	//neqn = number of electronic states
	//initially 1 for each state
	std::vector<int> ntbf = std::vector<int>(neqn, 1);

	//container for the state populations of the tbf's in each state
	std::vector<std::vector<vec1x > > pt_vec = std::vector<std::vector<vec1x > >(neqn, std::vector<vec1x>(1, vec1x(nt, complexd(0.0,0.0))));

	std::vector<std::vector<vec1x > > pt_gain_vec = std::vector<std::vector<vec1x > > (neqn-1, std::vector<vec1x>(1, vec1x(nt, complexd(0.0,0.0))));

	//container for the norm of all tbfs for each state, only vec<vec<>> for write_data_spawn function
	std::vector<vector<double> > normt_vec = std::vector<vector<double> >(1, std::vector<double>(nt, 0.0));

	//containers for the properties of each tbf of each state 
	vector<vector<vector<double> > > bondl_vec   = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > energy_vec  = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > kenergy_vec = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > charge1_vec = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > charge2_vec = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > spin1_vec   = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));
	vector<vector<vector<double> > > spin2_vec   = vector<vector<vector<double> > >(neqn, vector<vector<double>>(1, vector<double>(nt, 0.0)));

	//energy of the tbf in each state to be used in the RK4 procedure
	vector<vector<double> > E_spawn  = vector<vector<double> >(neqn, vector<double>(1));
	//bondlength index of each tbf in each state that determines the inital bondlength of a new tbf
	vector<vector<int> > iR_spawn    = vector<vector<int> >(neqn, vector<int>(1));	

	//Object for RK4 and population loss functions
	EOMDRIVER DRIVEEOM;
	DRIVEEOM.auger_gamma         = total_auger_gamma;
	DRIVEEOM.partial_auger_gamma = partial_auger_gamma;

	//vector time loop index at which spawning occurs
	std::vector<int> i_spawn;

	//number of spawning events
	int n_spawn = 0;
	int n_spawn_2 = 0;
	//max number of spawning events
	int n_spawn_max = nt / nt_spawn;

	//vector of amplitudes to be used in the RK4 proceudre
	std::vector<vec1x > y = std::vector<vec1x > (neqn, vec1x(1, complexd(0.0,0.0)));
	y[0][0] = 1.0;	
	int ntbf_prev;
	for (std::size_t i = 0; i < time.size(); ++i) {

		//scale time scale down so it numerically agrees with RK4
		//double t0 = tstart + i*(dt/10000);
		//double tf = t0 + dt/10000;
		
		double t0 = tstart + i*(dt);
		double tf = t0 + dt;
		
		//assign energy from data file at t to E_spawn for each trajectory
		//int tbf_count = 0;
		for (int j = 0; j<neqn; j++) {
			for (int k = 0; k<ntbf[j]; k++) {
				E_spawn[j][k]  = energy[j][i]; 
				//E_spawn[j][k]  = 0.0; 
				//tbf_count++;
			}
		}
		
		//pass energies for each tbf of each state
		DRIVEEOM.E_spawn = E_spawn;
		//number of states
		DRIVEEOM.neqn = neqn;
		//vector of number of tbfs per state
		DRIVEEOM.ntbf = ntbf;

		//run RK4 for spawning
		bool spawn = true;
		for (int j = 0; j < neqn; j++) {
			//total number of tbfs across all states
			DRIVEEOM.n = ntbf[j];	
			DRIVEEOM.neqn_index = j;
			DRIVEEOM.RK4(y[j], t0, tf, spawn);	
		}
		//cout << y[0][0].real() << " " << y[0][0].imag() << endl;

		//set norm at ti to 0
		normt_vec[0][i]  = 0.0;

		//assign amplitudes in y to population container vec
		for (int j = 0; j<neqn; j++) {
			for (int k = 0; k<ntbf[j]; k++) {
				pt_vec[j][k][i]    = std::norm(y[j][k]);
			}
		}
	
		//cout << i << " " << pt_vec[1][0][i].real() << endl;
		//
		int neqn_count = 1;
		if ((i > 0) && (i % nt_spawn == 0) && (n_spawn < n_spawn_max)) {
			n_spawn_2++;
			//cout << n_spawn_2 << endl;
			for (int j = 0; j < nstep; j++) {
				for (int k = 0; k < nchan; k++) {
					if (j > 0) {
						if (n_spawn_2 > 1) { 
							pt_gain_vec[neqn_count-1].resize(pt_vec[neqn_count-nchan].size()+1);
							pt_gain_vec[neqn_count-1][n_spawn_2-1] = vec1x(nt, complexd(0.0,0.0));
						}
					
					}
					neqn_count++;
				}
			}
		}
		
		neqn_count = 1;
		for (int j = 0; j < nstep; j++) {
			for (int k = 0; k < nchan; k++) {

				if (nstep == 1) {//pt_gain does not need to increase in size for single step decay
					//DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[0][0][i],pt_gain_vec[k][0][i],pt_gain_vec[k][0][i-1]);
					DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt,nt,pt_vec[0][0][i],pt_gain_vec[k][0][i],pt_gain_vec[k][0][i-1]);
				}
				else {
					if (j == 0 ) {
						//DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[0][0][i],pt_gain_vec[k][0][i],pt_gain_vec[k][0][i-1]);
						DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt,nt,pt_vec[0][0][i],pt_gain_vec[k][0][i],pt_gain_vec[k][0][i-1]);
						
						//cout << pt_vec[0][0][i] << endl;

						if ((i > 0) && (i % nt_spawn == 0) && (n_spawn < n_spawn_max)) {	
							//cout <<	"n_spawn: " << n_spawn << endl;
							//cout <<	"n_spawn_2: " << n_spawn_2 << endl;
							//cout << "neqn_count: " << neqn_count << endl;
							//cout << pt_gain_vec[k].size() << endl;
						}
						/*if (n_spawn == 1) {
							DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[0][0][i],pt_gain_vec[k][0][i],pt_gain_vec[k][0][i-1]);
						}
						if (n_spawn > 1) {
							for (int n = 0; n < pt_gain_vec[k].size(); n++) {
								DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[0][0][i],pt_gain_vec[k][n][i],pt_gain_vec[k][n][i-1]);
							}
						}*/
					}
					if (j > 0) {
						if (n_spawn == 0) continue;
						if (n_spawn == 1) {
							//DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[neqn_count-nchan][0][i],pt_gain_vec[neqn_count-1][0][i],pt_gain_vec[neqn_count-1][0][i-1]);
							DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt,nt,pt_vec[neqn_count-nchan][0][i],pt_gain_vec[neqn_count-1][0][i],pt_gain_vec[neqn_count-1][0][i-1]);
						}
						if (n_spawn > 1) {
						//cout << neqn_count-1 << endl;
							if ((i > 0) && (i % nt_spawn == 0) && (n_spawn < n_spawn_max)) {
										//cout <<	"n_spawn: " << n_spawn << endl;
										//cout <<	"n_spawn_2: " << n_spawn_2 << endl;
										//cout << "neqn_count: " << neqn_count << endl;
							}
							for (int n = 0; n < n_spawn; n++) {
								//DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt/10000,nt,pt_vec[neqn_count-nchan][n][i],pt_gain_vec[neqn_count-1][n][i],pt_gain_vec[neqn_count-1][n][i-1]);
								DRIVEEOM.Numerical_Population_Loss_Spawn(i,j,k,dt,nt,pt_vec[neqn_count-nchan][n][i],pt_gain_vec[neqn_count-1][n][i],pt_gain_vec[neqn_count-1][n][i-1]);
								//if ((i > 0) && (i % nt_spawn == 0) && (n_spawn < n_spawn_max)) {
									//if (neqn_count == 1) {
										//cout << n << endl;
										//pt_gain_vec[neqn_count-1][n][i];
									//}
								//}
								//std::cout << pt_gain_vec[neqn_count-1][n][i] << endl;
								//std::cout << pt_gain_vec[neqn_count-1][n][i-1] << endl;  
								//std::cout << y[neqn_count][n] << endl;	

							}
						}
					}
				}
				neqn_count++;
			}
		}

		//cout << pt_gain_vec[3][0][i] << endl;

		int nspawn_index_count = 0;			
		//if it is spawning time
		if ((i > 0) && (i % nt_spawn == 0) && (n_spawn < n_spawn_max)) {
			//log the time indexes where spawning occurs
			i_spawn.push_back(i);
			//cout << i << endl;	
			int y_index;
			//first spawn do nott need to increase any array size 
			if (n_spawn == 0) {

				for (int k = 0; k < nchan; k++) {
					
					//first spawn will just populated the spaces assigned by neqn
					y[k+1][0] = sqrt(pt_gain_vec[k][0][i]);
					
					//phase matching
					if (y[0][0].real() > 0) {
						y[k+1][0].real(y[k+1][0].real() * 1.0);  
					}
					else if (y[0][0].real() < 0) {
						y[k+1][0].real(y[k+1][0].real() * -1.0);   
					}
					if (y[0][0].imag() > 0) {
						y[k+1][0].imag(y[k+1][0].imag() * 1.0);   
					}
					else if (y[0][0].imag() < 0) {
						y[k+1][0].imag(y[k+1][0].imag() * -1.0);	
					}
				}
			}

			if (n_spawn > 0) {
				//second spawn (only increase size of states populated in the first spawning event.)
				neqn_count = 1;
				for (int j = 0; j < nstep; j++) {
					for (int k = 0; k < nchan; k++) {
				
						if (nstep == 1) {

							ntbf[k+1]++;
							y[k+1].resize(ntbf[k+1]);

							pt_vec[k+1].resize(ntbf[k+1]);
							pt_vec[k+1][n_spawn] = vec1x(nt, complexd(0.0,0.0));

							E_spawn[k+1].resize(ntbf[k+1]);
							iR_spawn[k+1].resize(ntbf[k+1]);
							
							y[k+1][n_spawn] = sqrt(pt_gain_vec[k][0][i]-pt_gain_vec[k][0][i_spawn[n_spawn-1]]);

							//phase matching
							if (y[0][0].real() > 0) {//parant trajectory
								y[k+1][n_spawn].real(y[k+1][n_spawn].real() * 1.0);  
							}
							else if (y[0][0].real() < 0) {
								y[k+1][n_spawn].real(y[k+1][n_spawn].real() * -1.0);   
							}
							if (y[0][0].imag() > 0) {
								y[k+1][n_spawn].imag(y[k+1][n_spawn].imag() * 1.0);   
							}
							else if (y[0][0].imag() < 0) {
								y[k+1][n_spawn].imag(y[k+1][n_spawn].imag() * -1.0);   
							}

							//bondlength and energy vector increase
							bondl_vec[k+1].resize(ntbf[k+1]);
							bondl_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
							energy_vec[k+1].resize(ntbf[k+1]);
							energy_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
							kenergy_vec[k+1].resize(ntbf[k+1]);
							kenergy_vec[k+1][n_spawn] = vector<double>(nt, 0.0);

							charge1_vec[k+1].resize(ntbf[k+1]);
							charge2_vec[k+1].resize(ntbf[k+1]);
							charge1_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
							charge2_vec[k+1][n_spawn] = vector<double>(nt, 0.0);

							spin1_vec[k+1].resize(ntbf[k+1]);
							spin2_vec[k+1].resize(ntbf[k+1]);
							spin1_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
							spin2_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
						}

						if (nstep > 1) {
							if (j == 0) {
								ntbf[k+1]++;
								y[k+1].resize(ntbf[k+1]);

								pt_vec[k+1].resize(ntbf[k+1]);
								pt_vec[k+1][n_spawn] = vec1x(nt, complexd(0.0,0.0));

								E_spawn[k+1].resize(ntbf[k+1]);
								iR_spawn[k+1].resize(ntbf[k+1]);
								/*	
								if (n_spawn == 1) {
									cout << pt_gain_vec[k][0][i] << endl;	
									cout << pt_gain_vec[k][0][i_spawn[n_spawn-1]] << endl;	
									cout << pt_gain_vec[k][0][i] - pt_gain_vec[k][0][i_spawn[n_spawn-1]] << endl;
									cout << i << endl;
									cout << i_spawn[n_spawn-1] << endl;
								}*/

								y[k+1][n_spawn] = sqrt(pt_gain_vec[k][0][i]-pt_gain_vec[k][0][i_spawn[n_spawn-1]]);

								//phase matching
								if (y[0][0].real() > 0) {//parant trajectory
									y[k+1][n_spawn].real(y[k+1][n_spawn].real() * 1.0);  
								}
								else if (y[0][0].real() < 0) {
									y[k+1][n_spawn].real(y[k+1][n_spawn].real() * -1.0);   
								}
								if (y[0][0].imag() > 0) {
									y[k+1][n_spawn].imag(y[k+1][n_spawn].imag() * 1.0);   
								}
								else if (y[0][0].imag() < 0) {
									y[k+1][n_spawn].imag(y[k+1][n_spawn].imag() * -1.0);   
								}

								//bondlength and energy vector increase
								bondl_vec[k+1].resize(ntbf[k+1]);
								bondl_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
								energy_vec[k+1].resize(ntbf[k+1]);
								energy_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
								kenergy_vec[k+1].resize(ntbf[k+1]);
								kenergy_vec[k+1][n_spawn] = vector<double>(nt, 0.0);

								charge1_vec[k+1].resize(ntbf[k+1]);
								charge2_vec[k+1].resize(ntbf[k+1]);
								charge1_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
								charge2_vec[k+1][n_spawn] = vector<double>(nt, 0.0);

								spin1_vec[k+1].resize(ntbf[k+1]);
								spin2_vec[k+1].resize(ntbf[k+1]);
								spin1_vec[k+1][n_spawn] = vector<double>(nt, 0.0);
								spin2_vec[k+1][n_spawn] = vector<double>(nt, 0.0);

							}
							if (j > 0) {
								if (n_spawn == 1) {
									
									y[neqn_count][0] = sqrt(pt_gain_vec[neqn_count-1][0][i]);
									
									//phase matching
									if (y[neqn_count-nchan][0].real() > 0) {
										y[neqn_count][0].real(y[neqn_count][0].real() * 1.0);  
									}
									else if (y[neqn_count-1][0].real() < 0) {
										y[neqn_count][0].real(y[neqn_count][0].real() * -1.0);   
									}
									if (y[neqn_count-1][0].imag() > 0) {
										y[neqn_count][0].imag(y[neqn_count][0].imag() * 1.0);   
									}
									else if (y[neqn_count-1][0].imag() < 0) {
										y[neqn_count][0].imag(y[neqn_count][0].imag() * -1.0);	
									}
									
								}
								if (n_spawn > 1) {
									if (k == 0) {
										nspawn_index_count += (n_spawn-1);
									}
									ntbf_prev = ntbf[neqn_count];	
									ntbf[neqn_count] += n_spawn;
								
									y[neqn_count].resize(ntbf[neqn_count]);

									pt_vec[neqn_count].resize(ntbf[neqn_count]);

									for (int n = ntbf_prev; n < ntbf[neqn_count]; n++) {
										pt_vec[neqn_count][n] = vec1x(nt, complexd(0.0,0.0));
									}

									E_spawn[neqn_count].resize(ntbf[neqn_count]);
									iR_spawn[neqn_count].resize(ntbf[neqn_count]);
									bondl_vec[neqn_count].resize(ntbf[neqn_count]);
									energy_vec[neqn_count].resize(ntbf[neqn_count]);
									kenergy_vec[neqn_count].resize(ntbf[neqn_count]);
									charge1_vec[neqn_count].resize(ntbf[neqn_count]);
									charge2_vec[neqn_count].resize(ntbf[neqn_count]);
									spin1_vec[neqn_count].resize(ntbf[neqn_count]);
									spin2_vec[neqn_count].resize(ntbf[neqn_count]);

									//cout << n_spawn << endl;	
									//for (int n = 0; n <ntbf[neqn_count-nchan]; n++) {
									int gain_index_count = 0;
									//cout << (neqn_count-1) << endl;
									//cout << pt_gain_vec[neqn_count-1].size() << endl;
									//cout << ntbf_prev << " " << ntbf[neqn_count] << endl;
									for (int n = ntbf_prev; n <ntbf[neqn_count]; n++) {
									
										y[neqn_count][n] = sqrt(pt_gain_vec[neqn_count-1][gain_index_count][i]-pt_gain_vec[neqn_count-1][gain_index_count][i_spawn[n_spawn-1]]);	
										//std::cout << n << endl;
										//std::cout << pt_gain_vec[neqn_count-1][gain_index_count][i] << endl;
										//std::cout << pt_gain_vec[neqn_count-1][gain_index_count][i_spawn[n_spawn-1]] << endl;  
										//std::cout << y[neqn_count][n] << endl;	
										//phase matching
										if (y[neqn_count-nchan][n-nspawn_index_count].real() > 0) {
											y[neqn_count][n].real(y[neqn_count][n].real() * 1.0);  
										}
										else if (y[neqn_count-nchan][n-nspawn_index_count].real() < 0) {
											y[neqn_count][n].real(y[neqn_count][n].real() * -1.0);   
										}
										if (y[neqn_count-nchan][n-nspawn_index_count].imag() > 0) {
											y[neqn_count][n].imag(y[neqn_count][n].imag() * 1.0);   
										}
										else if (y[neqn_count-nchan][n-nspawn_index_count].imag() < 0) {
											y[neqn_count][n].imag(y[neqn_count][n].imag() * -1.0);   
										}
											
										//bondlength and energy vector increase
										bondl_vec[neqn_count][n]   = vector<double>(nt, 0.0);
										energy_vec[neqn_count][n]  = vector<double>(nt, 0.0);
										kenergy_vec[neqn_count][n] = vector<double>(nt, 0.0);

										charge1_vec[neqn_count][n] = vector<double>(nt, 0.0);
										charge2_vec[neqn_count][n] = vector<double>(nt, 0.0);
										spin1_vec[neqn_count][n]   = vector<double>(nt, 0.0);
										spin2_vec[neqn_count][n]   = vector<double>(nt, 0.0);

										gain_index_count++;
										


									}

								}
							}
						}
						neqn_count++;
					}
				}
			}
			//determine closest bond length at t_spawn

			vector<double> R_diff;
			if (nstep == 1) {
				neqn_count = 1;
				for (int j = 0; j < nstep; j++) {
					for (int k = 0; k < nchan; k++) {
						R_diff = vector<double>(nt, 0.0);
						for (std::size_t t = 0; t < time.size(); ++t) {
								R_diff[t] = abs(bondl[0][i] - bondl[neqn_count][t]);	
						}
						auto it = std::min_element(std::begin(R_diff), std::end(R_diff));
						iR_spawn[neqn_count][n_spawn] = std::distance(std::begin(R_diff), it);
						neqn_count++;
					}
				}
			}
			if (nstep > 1) {
				neqn_count = 1;
				for (int j = 0; j < nstep; j++) {
					for (int k = 0; k < nchan; k++) {
						if (j == 0 ) {
							R_diff = vector<double>(nt, 0.0);
							for (std::size_t t = 0; t < time.size(); ++t) {
								R_diff[t] = abs(bondl[0][i] - bondl[neqn_count][t]);	
							}
							auto it = std::min_element(std::begin(R_diff), std::end(R_diff));
							iR_spawn[neqn_count][n_spawn] = std::distance(std::begin(R_diff), it);
							
						}
						if (j > 0) {
							if (n_spawn == 1) {
								R_diff = vector<double>(nt, 0.0);
								for (std::size_t t = 0; t < time.size(); ++t) {
									//R_diff[t] = abs(bondl_vec[neqn_count-nchan][n_spawn][i] - bondl[neqn_count][t]);
									R_diff[t] = abs(bondl_vec[neqn_count-nchan][0][i-1] - bondl[neqn_count][t]);	
								}
								auto it = std::min_element(std::begin(R_diff), std::end(R_diff));
								iR_spawn[neqn_count][0] = std::distance(std::begin(R_diff), it);
							}
							if (n_spawn > 1) {
								int gain_index_count = 0;
								for (int n = ntbf_prev; n <ntbf[neqn_count]; n++) {
									R_diff = vector<double>(nt, 0.0);
									for (std::size_t t = 0; t < time.size(); ++t) {
										R_diff[t] = abs(bondl_vec[neqn_count-nchan][gain_index_count][i-1] - bondl[neqn_count][t]);	
									}
									auto it = std::min_element(std::begin(R_diff), std::end(R_diff));
									iR_spawn[neqn_count][n] = std::distance(std::begin(R_diff), it);
									gain_index_count++;	
								}
							}
						}
						neqn_count++;
					}
				}
			}

			n_spawn += 1;
		}
	
		//cout << i << endl;
		for (int j = 0; j < neqn; j++) {
			for (int k = 0; k < ntbf[j]; k++) {
				
	   			pt_vec[j][k][i]    = std::norm(y[j][k]);
				if (j == 1) {
					//std::cout << k << endl;	
					//std::cout << std::norm(y[j][k]) << endl;
				}

			}
		}
	
		bondl_vec[0][0][i]   = bondl[0][i];
		energy_vec[0][0][i]  = energy[0][i];
		kenergy_vec[0][0][i] = kenergy[0][i];
		charge1_vec[0][0][i] = charge[0][0][i];
		charge2_vec[0][0][i] = charge[1][0][i];
		spin1_vec[0][0][i]   = spin[0][0][i];
		spin2_vec[0][0][i]   = spin[1][0][i];

		neqn_count = 1;
		for (int j = 0; j<nstep; j++) {
			for (int k = 0; k<nchan; k++) {
				for (int n = 0; n < ntbf[neqn_count]; n++) {

					if (j == 0 && n_spawn > 0) {
						bondl_vec[neqn_count][n][i]   = bondl[neqn_count][iR_spawn[neqn_count][n]];
						energy_vec[neqn_count][n][i]  = energy[neqn_count][iR_spawn[neqn_count][n]];
						kenergy_vec[neqn_count][n][i] = kenergy[neqn_count][iR_spawn[neqn_count][n]];
						charge1_vec[neqn_count][n][i] = charge[0][neqn_count][iR_spawn[neqn_count][n]];
						charge2_vec[neqn_count][n][i] = charge[1][neqn_count][iR_spawn[neqn_count][n]];
						spin1_vec[neqn_count][n][i]   = spin[0][neqn_count][iR_spawn[neqn_count][n]];
						spin2_vec[neqn_count][n][i]   = spin[1][neqn_count][iR_spawn[neqn_count][n]];
					}
					if (j > 0 && n_spawn > 1) {
						bondl_vec[neqn_count][n][i]   = bondl[neqn_count][iR_spawn[neqn_count][n]];
						energy_vec[neqn_count][n][i]  = energy[neqn_count][iR_spawn[neqn_count][n]];
						kenergy_vec[neqn_count][n][i] = kenergy[neqn_count][iR_spawn[neqn_count][n]];
						charge1_vec[neqn_count][n][i] = charge[0][neqn_count][iR_spawn[neqn_count][n]];
						charge2_vec[neqn_count][n][i] = charge[1][neqn_count][iR_spawn[neqn_count][n]];
						spin1_vec[neqn_count][n][i]   = spin[0][neqn_count][iR_spawn[neqn_count][n]];
						spin2_vec[neqn_count][n][i]   = spin[1][neqn_count][iR_spawn[neqn_count][n]];
					}
					iR_spawn[neqn_count][n]++;
				}
				neqn_count++;
			}
		}
		/*for (int j = 0; j<neqn; j++) {
			for (int k = 0; k<ntbf[j]; k++) {

				if (j == 0) {
					bondl_vec[j][k][i]  = bondl[j][i];
					energy_vec[j][k][i] = energy[j][i];
					charge1_vec[j][k][i] = charge[0][j][i];
					charge2_vec[j][k][i] = charge[1][j][i];
				}
				else if (j > 0 && n_spawn > 0) {
					bondl_vec[j][k][i] = bondl[j][iR_spawn[j][k]];
					energy_vec[j][k][i] = energy[j][iR_spawn[j][k]];
					charge1_vec[j][k][i] = charge[0][j][iR_spawn[j][k]];
					charge2_vec[j][k][i] = charge[1][j][iR_spawn[j][k]];
				}
				iR_spawn[j][k]++;
			}
		}*/

		for (int j = 0; j < neqn; j++) {                        
			for (int k = 0; k<ntbf[j]; k++) {
				normt_vec[0][i] += pt_vec[j][k][i].real();
			}
		}
	}

	FILEWRITER WRITEFILES;
	WRITEFILES.nt = nt;
	WRITEFILES.n_print = 1;
	WRITEFILES.neqn = neqn;
	WRITEFILES.tf_vec = time;
	WRITEFILES.write_data_files_spawn(pt_vec, normt_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("bondlength", bondl_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("energy", energy_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("kenergy", kenergy_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("0_charge", charge1_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("1_charge", charge2_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("0_spin",   spin1_vec, ntbf);
	WRITEFILES.write_data_files_spawn_dat("1_spin",   spin2_vec, ntbf);

}



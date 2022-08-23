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
#ifndef PULSE_INTERACTION_H
#define PULSE_INTERACTION_H

class TDSEUTILITY {

public:

	TDSEUTILITY();
	~TDSEUTILITY();

	int fv_sample;
	int bw_sample;
	int neqn;
	int nt;
	double tstart;
	double dt;
	vector<double> tmax;
	vector<vector<vector<double> > > field_strength;
	vector<vector<double> > gw;
	vector<vector<double> > wn;
	vector<double> var;
	vector<vector<double> > wx;
	vector<arma::mat> Matrix;
	vector<vector<double> > mu;
	vector<vector<vector<double> > > decay_widths;
	vector<string> decay_channels;
	vector<bool> BOOL_VEC;
	int n_pulse;


double icalib(vector<arma::mat> Matrix, vector<vector<double> > mu, std::vector<vector<double> > ww, double spot_size, vector<double> var);

void focal_volume_average(double spot_size, vector<vector<double> >& field_strength, vector<double> intensity, int shell_sample);

void bandwidth_average(double bw, std::vector<vector<double> >& gw, std::vector<vector<double> >& wn, std::vector<double> wx, int bw_sample, int bw_extent);

void eom_run(int ei, vector<double>& tf_vec, vector<vec1x >& pt_vec_avg, vector<double>& norm_t_vec_avg);

private:
double gaussian_field(double t, double t_max, double amp_max, double var);

};
#endif

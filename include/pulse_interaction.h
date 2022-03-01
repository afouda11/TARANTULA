#ifndef PULSE_INTERACTION_H
#define PULSE_INTERACTION_H

class TDSEUTILITY {

public:

TDSEUTILITY(void);

	int shell_sample;
	int band_sample;
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


double icalib(vector<arma::mat> Matrix, vector<vector<double> > mu, std::vector<vector<double> > ww, double spot_size, vector<double> var);

void focal_volume_average(double spot_size, vector<vector<double> >& field_strength, vector<double> intensity, int shell_sample);

void bandwidth_average(double bw, std::vector<vector<double> >& gw, std::vector<vector<double> >& wn, std::vector<double> wx, vector<double> bandwidth_avg);

void eom_run(int ei, vector<double>& tf_vec, vector<vec1x >& pt_vec_avg, vector<double>& norm_t_vec_avg);

private:
double gaussian_field(double t, double t_max, double amp_max, double var);

};
#endif

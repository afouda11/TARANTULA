#ifndef PULSE_INTERACTION_H
#define PULSE_INTERACTION_H

double gaussian_field(double t, double t_max, double amp_max, double var);

void focal_volume_average(double spot_size, vector<vector<double> >& field_strength, vector<double> intensity, int shell_sample);

void bandwidth_average(double bw, std::vector<vector<double> >& gw, std::vector<vector<double> >& wn, std::vector<double> wx, int band_sample);

void rk4_run(int ei, int shell_sample, int band_sample, int neqn, int nval, int nt, double tstart, double dt,
vector<double> tmax, vector<vector<vector<double> > > field_strength, vector<vector<double> > gw, vector<vector<double>
> wn, vector<double> var, vector<vector<double> > wx, vector<arma::mat> Matrix, vector<vector<double> > polarization,
vector<vector<vector<double> > > decay_widths, bool RWA, bool ECALC, bool DECAY, bool TWOPULSE, bool GAUSS, bool
BANDW_AVG, bool STARK, bool WRITE_PULSE, vector<double>& tf_vec, vector<vec1x >& pt_vec_avg, vector<double>& norm_t_vec_avg);

#endif

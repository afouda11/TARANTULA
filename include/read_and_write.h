#ifndef READ_AND_WRITE_H
#define READ_AND_WRITE_H

#include <armadillo>
#include <iostream>
#include <iterator>
#include "vectypedef.hpp"

template <typename T>
void file2vector(std::string filename, std::vector<T>& vec) {
    std::ifstream file(filename);
    std::istream_iterator<T> start(file);
    std::istream_iterator<T> end;
    copy(start, end, back_inserter(vec));
    return;
}

bool read_bool_options(string option);

arma::mat read_matrix(int n, std::string Diagonal, std::string Off_Diagonal);

void group_sum(int n_type, int nt, int neqn, int n_decay_chan, vector<vec1x > & pt_sum_vec, vector<vec1x > pt_vec);

void pair_sum(int nt, int n_type, int n_decay_chan, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_);

class FILEWRITER {

public:

    int nt;
    int n_type;
    int n_sum_type;
    int n_print;
    vector<double> tf_vec;

FILEWRITER(void);

void write_data_files(string outfilename, vector<vec1x> pt_vec, vector<vec1x> pt_sum_vec, vector<vec1x>& pt_vec_perp, vector<vec1x>& pt_sum_vec_perp, vector<double> norm_t_vec_avg, vector<double>& norm_t_vec_avg_perp, bool SUM, bool PERP_AVG);

void write_data_variable_files(int n_photon_e, int n_calc, vector<double> intensity, vector<double> wx, vector<vector<vec1x> > pt_vec, vector<vector<vec1x> > pt_sum_vec, vector<vector<vec1x> > pt_vec_perp, vector<vector<vec1x> > pt_sum_vec_perp, bool SUM, bool ECALC, bool PERP_AVG);

private:

void write_data(string outfilename, int neqn, vector<vec1x > pt_vec, vector<double> norm_t_vec, bool GNUPLOT_OUT); 

void write_data_variable(string outfilename, int n_calc, int neqn, vector<double> variable, vector<vector<vec1x > > pt_vec, bool GNUPLOT_OUT);

};

void write_field(string outfilename, int nt, int n_print, vector<double> tf_vec, vector<double> field);

string convertInt(int number);

#endif

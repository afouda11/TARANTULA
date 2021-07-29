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

void write_data(std::string outfilename, int nt, int neqn, int n_print, std::vector<double> tf_vec, std::vector<vec1x > pt_vec, std::vector<double> norm_t_vec, bool GNUPLOT_OUT); 

void write_data_variable(std::string outfilename, int n_calc, int nt, int neqn, std::vector<double> variable, std::vector<vector<vec1x > > pt_vec, bool GNUPLOT_OUT);

void write_field(std::string outfilename, int nt, int n_print, std::vector<double> tf_vec, std::vector<double> field);

void group_sum(int n_type, int nt, int neqn, vector<vec1x > & pt_sum_vec, vector<vec1x > pt_vec, int model);

void degenerate_pair_sum(int nt, int model, int edge, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_);

string convertInt(int number);

#endif

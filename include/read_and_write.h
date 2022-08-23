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
#ifndef READ_AND_WRITE_H
#define READ_AND_WRITE_H

#include <armadillo>
#include <iostream>
#include <iterator>
#include <sstream>
#include "vectypedef.hpp"

template <typename T>
void file2vector(std::string filename, std::vector<T>& vec) {
    std::ifstream file(filename.c_str());
    std::istream_iterator<T> start(file);
    std::istream_iterator<T> end;
    copy(start, end, back_inserter(vec));
    return;
}
template <typename T>
void read_options(string option, T & result) {
    string line;
    ifstream myfile ("inputs/input.dat");
    vector<string> inputs;
    if (myfile.is_open()) {
        while ( getline (myfile,line) ) {
            istringstream iss(line);
            copy(istream_iterator<string>(iss),
            istream_iterator<string>(),
            back_inserter(inputs));
        }
        myfile.close();
    }
	int i = 0;
	stringstream ss;
    for (vector<string>::iterator t=inputs.begin(); t!=inputs.end(); t++) {
        if(*t == option) {
			ss << inputs.at(i+1);
			ss >> result;
        }
		i++;
    }     
    return;
}
bool read_bool_options(string option);

arma::mat read_matrix(int n, std::string Diagonal, std::string Off_Diagonal);

void group_sum(int n_type, int nt, int neqn, int n_decay_chan, vector<vec1x > & pt_sum_vec, vector<vec1x > pt_vec);

void pair_sum(int nt, int n_type, int n_decay_chan, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_);

class FILEWRITER {

public:

	FILEWRITER(); 
	~FILEWRITER(); 

	int nt;
    int n_print;
    vector<double> tf_vec;
	vector<bool> BOOL_VEC;
	int n_photon_e;
	int n_calc;
	int neqn;
	int orient_avg;
	vector<double> variable;
	string varstring;

void write_data_files(string outfilename, vector<vector<vec1x> >& pt_vec, vector<vector<double> >& normt_vec);

void write_data_variable_files(vector<vector<vector<vec1x> > > pt_vec);

private:

void write_data(string outfilename, int ncol, vector<vec1x > pt_vec, vector<double> norm_t_vec, bool GNUPLOT_OUT); 

void write_data_variable(string outfilename, int ncol, vector<vector<vector<vec1x > > > pt_vec, int mu, bool GNUPLOT_OUT);

};

void write_field(string outfilename, int nt, int n_print, vector<double> tf_vec, vector<double> field);

string convertInt(int number);

#endif

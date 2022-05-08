#include <iostream>
#include <armadillo>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
#include <sstream>
#include "vectypedef.hpp"
#include "read_and_write.h"

bool read_bool_options(string option) {
    bool result = false;
    string line;
    ifstream myfile ("inputs/input.dat");
    vector<string> bools;
    if (myfile.is_open()) {
        while ( getline (myfile,line) ) {
            istringstream iss(line);
            copy(istream_iterator<string>(iss),
            istream_iterator<string>(),
            back_inserter(bools));
        }
        myfile.close();
    }
	int i = 0;
    for (vector<string>::iterator t=bools.begin(); t!=bools.end(); t++) {
        if(*t == option) {
            if(bools.at(i+1) == "false") {
                result          = false;
            }
            if(bools.at(i+1) == "true") {
                result          = true;
            }    
        }
		i++;
    }     
    return result;
}


arma::mat read_matrix(int n, std::string Diagonal, std::string Off_Diagonal) {

    std::ifstream file(Diagonal.c_str());	
	std::istream_iterator<double> start(file), end;
	std::vector<double> diagonal(start, end);	
	
	arma::Col<double> diag(n);
	for(int i = 0; i < n; i++) {	
		diag[i] = diagonal[i];
	}

 	int	col1, col2;
	double col3;
	std::vector<int>	ivec;
	std::vector<int>	jvec;	
	std::vector<double>	val;
	std::ifstream in(Off_Diagonal.c_str());
	while(!in.eof()){
		in >> col1;
  		ivec.push_back(col1);
		in >> col2;
  		jvec.push_back(col2);
		in >> col3;
  	 	val.push_back(col3);
	}

	arma::mat STATES = arma::diagmat(diag);
	size_t v  = ivec.size();
	for(size_t k = 0; k < v; k++) {

		STATES(ivec[k],jvec[k]) = val[k];	
		STATES(jvec[k],ivec[k]) = val[k];

	}
  
	return STATES;
}

//sums groups of states indcated by index files
void group_sum(int n_sum_type, int nt, int neqn, int n_decay_chan, vector<vec1x >& pt_sum_vec, vector<vec1x > pt_vec) {

    std::vector<string> group_names;
    file2vector("inputs/group_names.txt", group_names);
    std::vector<vector<int> > groups;
    groups = vector<vector<int> >((n_sum_type-(n_decay_chan+1))/(n_decay_chan+1));

    for (int i = 0; i < (n_sum_type-(n_decay_chan+1))/(n_decay_chan+1); i++) {
        file2vector("inputs/"+group_names[i]+"_index.txt",  groups[i]); 
    }
    int n_state = neqn / (n_decay_chan+1);
    for (int i = 0; i<nt; i++) {
        for (int n = 0; n < n_decay_chan+1; n++) {
            pt_sum_vec[0 + (n*(n_sum_type /(n_decay_chan+1)))][i] = pt_vec[0 + (n*n_state)][i];
            for (int j = 1; j<n_state; j++) {
                for (int k = 0; k < (n_sum_type-(n_decay_chan+1))/(n_decay_chan+1); k++) {//v fiddly (want k to index over group_names)
                                                                                          //the fact that the ground
                                                                                          //isnt included in group_names
                                                                                          //makes it more weird
                    for (int l = 0; l < static_cast<int>(groups[k].size()); l++) {
                            
                        if (j == groups[k][l]) {

                            pt_sum_vec[(k+1) + (n*(n_sum_type/(n_decay_chan+1)))][i] += pt_vec[j + (n*n_state)][i];

                        }

                    }
                }
            }
        }         
    }      
    return;
}               

void pair_sum(int nt, int n_type, int n_decay_chan, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_)
{
 	int	col1, col2;
	std::vector<int>	one;
	std::vector<int>	two;	
	std::ifstream in("inputs/state_pairs.txt");
	while(!in.eof()){
		in >> col1;
  		one.push_back(col1);
		in >> col2;
  		two.push_back(col2);
	}
    int n_actual_type = n_type / (n_decay_chan+1);
    int neqn = static_cast<int>(pt_vec_avg_.size());
    int n_state = neqn / (n_decay_chan+1);

    for(int i = 0; i<nt; i++) {
        for(int n = 0; n < n_decay_chan+1; n++) {
            for(int j = 0; j < n_actual_type; j++) {  
                pt_vec_[j + (n*n_actual_type)][i] = pt_vec_avg_[one[j] + (n*n_state)][i];
                if (two[j] > 0) {
                    pt_vec_[j + (n*n_actual_type)][i] += pt_vec_avg_[two[j] + (n*n_state)][i];

                }      
            }      
        }
    }
    return;
}

FILEWRITER::FILEWRITER(){
}
FILEWRITER::~FILEWRITER(){
}

void FILEWRITER::write_data_files(string outfilename, vector<vector<vec1x> >& pt_vec, vector<vector<double> >& normt_vec) {
   
    string outfilepath = "outputs/"+outfilename;
	if (orient_avg == 1) {
    	write_data(outfilepath+".txt", neqn, pt_vec[0], normt_vec[0], false);
	}
	if (orient_avg == 2) {
        write_data(outfilepath+"_x.txt", neqn, pt_vec[0], normt_vec[0], false);
        write_data(outfilepath+"_y.txt", neqn, pt_vec[1], normt_vec[1], false);
        for (int i = 0; i<nt; i++) {
            for (int j = 0; j<neqn; j++) {
                 pt_vec[2][j][i] = (pt_vec[0][j][i] + pt_vec[1][j][i]) / 2.0;
            }
            normt_vec[2][i] = (normt_vec[0][i] + normt_vec[1][i]) /2.0;
        }
        write_data(outfilepath+"_perp_avg.txt", neqn, pt_vec[2], normt_vec[2], false);         
    }
	if (orient_avg == 3) {
        write_data(outfilepath+"_x.txt", neqn, pt_vec[0], normt_vec[0], false);
        write_data(outfilepath+"_y.txt", neqn, pt_vec[1], normt_vec[1], false);
        write_data(outfilepath+"_z.txt", neqn, pt_vec[2], normt_vec[1], false);
        for (int i = 0; i<nt; i++) {
            for (int j = 0; j<neqn; j++) {
                 pt_vec[3][j][i] = (pt_vec[0][j][i] + pt_vec[1][j][i] + pt_vec[2][j][i]) / 3.0;
            }
            normt_vec[3][i] = (normt_vec[0][i] + normt_vec[1][i] + normt_vec[1][i]) / 3.0;
        }
        write_data(outfilepath+"_oreint_avg.txt", neqn, pt_vec[3], normt_vec[3], false);         
    }
}


void FILEWRITER::write_data_variable_files(vector<vector<vector<vec1x> > > pt_vec) {

	if (varstring == "energy") { //convert energy to ev for printing
		for (int i = 0; i < n_photon_e; i++) variable[i] *=  27.2114;
	}
    
    cout << "Writing energy/intensity variable data at t_final\n" << endl;
    string outfilepath = "outputs/";
	if (orient_avg == 1) {
		write_data_variable(outfilepath+""+varstring+"_variable.txt", neqn, pt_vec, 0, false);
	}
	if (orient_avg == 2) {
		write_data_variable(outfilepath+""+varstring+"_variable_x.txt", neqn, pt_vec, 0, false);
		write_data_variable(outfilepath+""+varstring+"_variable_y.txt", neqn, pt_vec, 1, false);
		write_data_variable(outfilepath+""+varstring+"_variable_perp.txt", neqn, pt_vec, 2, false);
	}
	if (orient_avg == 3) {
		write_data_variable(outfilepath+""+varstring+"_variable_x.txt", neqn, pt_vec, 0, false);
		write_data_variable(outfilepath+""+varstring+"_variable_y.txt", neqn, pt_vec, 1, false);
		write_data_variable(outfilepath+""+varstring+"_variable_z.txt", neqn, pt_vec, 2, false);
		write_data_variable(outfilepath+""+varstring+"_variable_orient.txt", neqn, pt_vec, 3, false);
	}
}    

void FILEWRITER::write_data(string outfilename, int ncol, vector<vec1x > pt_vec, vector<double> norm_t_vec, bool GNUPLOT_OUT) {
    
    //Write data to files
    std::streambuf *coutbuf = std::cout.rdbuf();; //save old buf
    string outfile = outfilename;
    std::ofstream out(outfile.c_str());
    std::cout.rdbuf(out.rdbuf());
    if (!GNUPLOT_OUT) {
        for(int i = 0; i<nt; i++) {
            if (i % n_print == 0) cout<< tf_vec[i] <<" ";
            for (int j = 0; j<ncol; j++) if (i % n_print == 0) cout << pt_vec[j][i].real() <<" ";
            if (i % n_print == 0) cout<< norm_t_vec[i] <<"\n";
        }
    }
    if (GNUPLOT_OUT) {
        for (int j = 0; j<ncol; j++) {
            for(int i = 0; i<nt; i++) if (i % n_print == 0) cout << tf_vec[i] <<" " << pt_vec[j][i].real() <<"\n ";
            cout << "\n " << "\n ";
        }
        for(int i = 0; i<nt; i++) {
            if (i % n_print == 0) cout << tf_vec[i] <<" " << norm_t_vec[i] <<"\n";
        }
    }

    std::cout.rdbuf(coutbuf); //reset to standard output again
    return;
}

void FILEWRITER::write_data_variable(string outfilename, int ncol, vector<vector<vector<vec1x > > > pt_vec, int mu, bool GNUPLOT_OUT) {

    //only works on avaraged over the orientation data
    std::streambuf *coutbuf = std::cout.rdbuf();; //save old buf
    string outfile = outfilename;
    std::ofstream out(outfile.c_str());
    std::cout.rdbuf(out.rdbuf());
    if (!GNUPLOT_OUT) {
        for(int i = 0; i<n_calc; i++) {
            cout<< variable[i] <<" ";
            for (int j = 0; j<ncol; j++) cout << pt_vec[i][mu][j][nt-1].real() <<" ";
            cout <<"\n";
        }
    }      
    if (GNUPLOT_OUT) {           
        for (int j = 0; j<ncol; j++) {
            for(int i = 0; i<n_calc; i++) cout << variable[i] <<" " << pt_vec[i][mu][j][nt-1].real() <<"\n ";
            cout << "\n " << "\n ";
        }
    }       
    std::cout.rdbuf(coutbuf); //reset to standard output again       
    return;

}       

void write_field(string outfilename, int nt, int n_print, vector<double> tf_vec, vector<double> field) {
    
    //Write data to files
    std::streambuf *coutbuf = std::cout.rdbuf();; //save old buf
    string outfile = outfilename;
    std::ofstream out(outfile.c_str());
    std::cout.rdbuf(out.rdbuf());
    for(int i = 0; i<nt; i++) if (i % n_print == 0) cout << tf_vec[i] <<" " << field[i] <<"\n ";
    cout << "\n " << "\n ";

    std::cout.rdbuf(coutbuf); //reset to standard output again
    return;

}


string convertInt(int number)
{
    stringstream n;
    n << number;
    return n.str();
}

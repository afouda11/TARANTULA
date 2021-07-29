#include <iostream>
#include <armadillo>
#include <iterator>
#include <vector>
#include <algorithm>
#include <cmath>

#include "vectypedef.hpp"
#include "read_and_write.h"


bool read_bool_options(string option) {

    bool result = false;
    string line;
    ifstream myfile ("inputs/bools.txt");
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

    std::ifstream file(Diagonal);	
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
	std::ifstream in(Off_Diagonal);
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

void write_data(std::string outfilename, int nt, int neqn, int n_print, std::vector<double> tf_vec, std::vector<vec1x > pt_vec, std::vector<double> norm_t_vec, bool GNUPLOT_OUT) {
    
    //Write data to files
    std::streambuf *coutbuf = std::cout.rdbuf();; //save old buf
    string outfile = outfilename;
    std::ofstream out(outfile.c_str());
    std::cout.rdbuf(out.rdbuf());
    if (!GNUPLOT_OUT) {
        for(int i = 0; i<nt; i++) {
            if (i % n_print == 0) cout<< tf_vec[i] <<" ";
            for (int j = 0; j<neqn; j++) if (i % n_print == 0) cout << pt_vec[j][i].real() <<" ";
            if (i % n_print == 0) cout<< norm_t_vec[i] <<"\n";
        }
    }
    if (GNUPLOT_OUT) {
        for (int j = 0; j<neqn; j++) {
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

void write_data_variable(std::string outfilename, int n_calc, int nt, int neqn, std::vector<double> variable, std::vector<vector<vec1x > > pt_vec, bool GNUPLOT_OUT) {

    //only works on avaraged over the orientation data
    std::streambuf *coutbuf = std::cout.rdbuf();; //save old buf
    string outfile = outfilename;
    std::ofstream out(outfile.c_str());
    std::cout.rdbuf(out.rdbuf());
    if (!GNUPLOT_OUT) {
        for(int i = 0; i<n_calc; i++) {
            cout<< variable[i] <<" ";
            for (int j = 0; j<neqn; j++) cout << pt_vec[i][j][nt-1].real() <<" ";
            cout <<"\n";
        }
    }      
    if (GNUPLOT_OUT) {           
        for (int j = 0; j<neqn; j++) {
            for(int i = 0; i<n_calc; i++) cout << variable[i] <<" " << pt_vec[i][j][nt-1].real() <<"\n ";
            cout << "\n " << "\n ";
        }
    }       
    std::cout.rdbuf(coutbuf); //reset to standard output again       
    return;

}       

void write_field(std::string outfilename, int nt, int n_print, std::vector<double> tf_vec, std::vector<double> field) {
    
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

//sums groups of states indcated by index files
void group_sum(int n_type, int nt, int neqn, vector<vec1x >& pt_sum_vec, vector<vec1x > pt_vec, int model) {
 
    if(model == 0) {
        
        //final valence state index
        std::vector<int> Fv_index;
        file2vector("inputs/Fv_index.txt", Fv_index);

        //final rydberg state index
        std::vector<int> Fr_index;
        file2vector("inputs/Fr_index.txt", Fr_index);

        //intermediate valence state index
        std::vector<int> Iv_index;
        file2vector("inputs/Iv_index.txt", Iv_index);

        //intermediate rydberg state index
        std::vector<int> Ir_index;
        file2vector("inputs/Ir_index.txt", Ir_index);

        //intermediate double state index
        std::vector<int> Id_index;
        file2vector("inputs/Id_index.txt", Id_index);
        cout << "Index's read" << endl;
    
        for(int i = 0; i<nt; i++) {
            for (int j = 0; j<neqn; j++) {

                pt_sum_vec[0][i] = pt_vec[0][i];

                for (int Fv = 0; Fv < static_cast<int>(Fv_index.size()); Fv++) {
                        
                    if (j == Fv_index[Fv]) {

                        pt_sum_vec[1][i] += pt_vec[j][i];

                    }

                }
                for (int Fr = 0; Fr < static_cast<int>(Fr_index.size()); Fr++) {
                        
                    if (j == Fr_index[Fr]) {

                        pt_sum_vec[2][i] += pt_vec[j][i];

                    }

                }
                for (int Iv = 0; Iv < static_cast<int>(Iv_index.size()); Iv++) {
                        
                    if (j == Iv_index[Iv]) {

                        pt_sum_vec[3][i] += pt_vec[j][i];

                    }

                }
                for (int Ir = 0; Ir < static_cast<int>(Ir_index.size()); Ir++) {
                        
                    if (j == Ir_index[Ir]) {

                        pt_sum_vec[4][i] += pt_vec[j][i];

                    }

                }
                for (int Id = 0; Id < static_cast<int>(Id_index.size()); Id++) {
                        
                    if (j == Id_index[Id]) {

                        pt_sum_vec[5][i] += pt_vec[j][i];
    
                    }
                }
            }
        }
    }
    if(model == 1) {
        //final valence state index
        std::vector<int> Fv_index;
        file2vector("inputs/Fv_index.txt", Fv_index);

        //intermediate valence state index
        std::vector<int> Iv_index;
        file2vector("inputs/Iv_index.txt", Iv_index);

        cout << "Index's read" << endl; 
        for(int i = 0; i<nt; i++) {
            for (int j = 0; j<neqn; j++) {

                pt_sum_vec[0][i] = pt_vec[0][i];

                for (int Fv = 0; Fv < static_cast<int>(Fv_index.size()); Fv++) {
                        
                    if (j == Fv_index[Fv]) {

                        pt_sum_vec[1][i] += pt_vec[j][i];

                    }

                }
                for (int Iv = 0; Iv < static_cast<int>(Iv_index.size()); Iv++) {
                        
                    if (j == Iv_index[Iv]) {

                        pt_sum_vec[2][i] += pt_vec[j][i];

                    }

                }
            }
        }
    }

    return;
}               

void degenerate_pair_sum(int nt, int model, int edge, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_)
{
    
        //sum the DEGENERATE pairs of pi-pi* val states only really relevant to nitric oxide
        for(int i = 0; i<nt; i++) {
            //ground state
          /*  pt_vec_[0][i]  = pt_vec_avg_[0][i];
            pt_vec_[1][i]  = pt_vec_avg_[1][i];
            pt_vec_[1][i] += pt_vec_avg_[2][i];
            pt_vec_[2][i]  = pt_vec_avg_[3][i];
            pt_vec_[2][i] += pt_vec_avg_[4][i];
            pt_vec_[3][i]  = pt_vec_avg_[5][i];
            pt_vec_[4][i]  = pt_vec_avg_[6][i];
*/
            pt_vec_[0][i]  = pt_vec_avg_[0][i];
            pt_vec_[1][i]  = pt_vec_avg_[1][i];
            pt_vec_[2][i]  = pt_vec_avg_[2][i];
            pt_vec_[2][i] += pt_vec_avg_[3][i];
            pt_vec_[3][i]  = pt_vec_avg_[4][i];
            pt_vec_[4][i]  = pt_vec_avg_[5][i];
        }      
    return;
}


string convertInt(int number)
{
    stringstream n;
    n << number;
    return n.str();
}

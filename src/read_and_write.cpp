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

//sums groups of states indcated by index files
void group_sum(int n_sum_type, int nt, int neqn, vector<vec1x >& pt_sum_vec, vector<vec1x > pt_vec) {

    std::vector<string> group_names;
    file2vector("inputs/group_names.txt", group_names);
    std::vector<vector<int> > groups;
    groups = vector<vector<int> >(n_sum_type-1);

    for (int i = 0; i < n_sum_type-1; i++) {
        file2vector("inputs/"+group_names[i]+"_index.txt",  groups[i]); 
    }

    for(int i = 0; i<nt; i++) {
        for (int j = 0; j<neqn; j++) {

            pt_sum_vec[0][i] = pt_vec[0][i];
            for (int k = 0; k < n_sum_type-1; k++) {   

                for (int l = 0; l < static_cast<int>(groups[k].size()); l++) {
                        
                    if (j == groups[k][l]) {

                        pt_sum_vec[k+1][i] += pt_vec[j][i];

                    }

                }
            }
        }
    }      

    return;
}               

void pair_sum(int nt, int n_type, vector<vec1x >& pt_vec_, vector<vec1x > pt_vec_avg_)
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

    for(int i = 0; i<nt; i++) {
        for(int j = 0; j < n_type; j++) {  
            pt_vec_[j][i] = pt_vec_avg_[one[j]][i];
            if (two[j] > 0) {
                pt_vec_[j][i] += pt_vec_avg_[two[j]][i];

            }      
        }      

    }
    return;
}

FILEWRITER::FILEWRITER(void){
}

void FILEWRITER::write_data_files(string outfilename, vector<vec1x> pt_vec, vector<vec1x> pt_sum_vec, vector<vec1x>& pt_vec_perp, vector<vec1x>& pt_sum_vec_perp, vector<double> norm_t_vec_avg, vector<double>& norm_t_vec_avg_perp, bool SUM, bool PERP_AVG) {
   
    if (SUM) {
        string gnufilepath = "outputs/gnu/sum/"+outfilename;        
        string outfilepath = "outputs/out/sum/"+outfilename;        
        write_data(gnufilepath+".txt", n_sum_type, pt_sum_vec, norm_t_vec_avg, true);
        write_data(outfilepath+".txt", n_sum_type, pt_sum_vec, norm_t_vec_avg, false);
        if(PERP_AVG) {
            write_data(gnufilepath+"_x.txt", n_sum_type, pt_sum_vec, norm_t_vec_avg, true);
            write_data(outfilepath+"_x.txt", n_sum_type, pt_sum_vec, norm_t_vec_avg, false);
            write_data(gnufilepath+"_y.txt", n_sum_type, pt_sum_vec_perp, norm_t_vec_avg_perp, true);
            write_data(outfilepath+"_y.txt", n_sum_type, pt_sum_vec_perp, norm_t_vec_avg_perp, false);
            for(int i = 0; i<nt; i++) {
                for (int j = 0; j<n_sum_type; j++) {
                  pt_sum_vec_perp[j][i] = (pt_sum_vec_perp[j][i] + pt_sum_vec[j][i]) / 2.0; 
                }
                norm_t_vec_avg_perp[i] = (norm_t_vec_avg_perp[i] + norm_t_vec_avg[i]) / 2.0;
            }
            write_data(gnufilepath+"_perp.txt", n_sum_type, pt_sum_vec_perp, norm_t_vec_avg_perp, true);
            write_data(outfilepath+"_perp.txt", n_sum_type, pt_sum_vec_perp, norm_t_vec_avg_perp, false);
        }
    }
    string gnufilepath = "outputs/gnu/"+outfilename;        
    string outfilepath = "outputs/out/"+outfilename;        
    write_data(gnufilepath+".txt", n_type, pt_vec, norm_t_vec_avg, true);
    write_data(outfilepath+".txt", n_type, pt_vec, norm_t_vec_avg, false);
    if(PERP_AVG) {
        write_data(gnufilepath+"_x.txt", n_type, pt_vec, norm_t_vec_avg, true);
        write_data(outfilepath+"_x.txt", n_type, pt_vec, norm_t_vec_avg, false);
        write_data(gnufilepath+"_y.txt", n_type, pt_vec_perp, norm_t_vec_avg_perp, true);
        write_data(outfilepath+"_y.txt", n_type, pt_vec_perp, norm_t_vec_avg_perp, false);
        for(int i = 0; i<nt; i++) {
            for (int j = 0; j<n_type; j++) {
                 pt_vec_perp[j][i] = (pt_vec_perp[j][i] + pt_vec[j][i]) / 2.0;
            }
            norm_t_vec_avg_perp[i] = (norm_t_vec_avg_perp[i] + norm_t_vec_avg[i]) /2.0;
        }
        write_data(gnufilepath+"_perp.txt", n_type, pt_vec_perp, norm_t_vec_avg_perp, true);
        write_data(outfilepath+"_perp.txt", n_type, pt_vec_perp, norm_t_vec_avg_perp, false);         
    }
}


void FILEWRITER::write_data_variable_files(int n_photon_e, int n_calc, vector<double> intensity, vector<double> wx, vector<vector<vec1x> > pt_vec, vector<vector<vec1x> > pt_sum_vec, vector<vector<vec1x> > pt_vec_perp, vector<vector<vec1x> > pt_sum_vec_perp, bool SUM, bool ECALC, bool PERP_AVG) {

//convert energy to ev for printing
for(int i = 0; i < n_photon_e; i++) wx[i] *=  27.2114;
    
    string gnufilepath = "outputs/gnu/sum/variable/";
    string outfilepath = "outputs/out/sum/variable/";
    cout << "Writing energy/intensity variable data at t_final\n" << endl;
    if(SUM and ECALC) {
        write_data_variable(gnufilepath+"photon_energy.txt", n_calc, n_sum_type, wx, pt_sum_vec, true); 
        write_data_variable(outfilepath+"photon_energy.txt", n_calc, n_sum_type, wx, pt_sum_vec, false); 
    }
    if(SUM and !ECALC) {
        write_data_variable(gnufilepath+"pulse_intensity.txt", n_calc, n_sum_type, intensity, pt_sum_vec, true); 
        write_data_variable(outfilepath+"pulse_intensity.txt", n_calc, n_sum_type, intensity, pt_sum_vec, false); 
    }
    gnufilepath = "outputs/gnu/variable/";
    outfilepath = "outputs/out/variable/";
    if(ECALC) {
        write_data_variable(gnufilepath+"photon_energy.txt", n_calc, n_type, wx, pt_vec, true); 
        write_data_variable(outfilepath+"photon_energy.txt", n_calc, n_type, wx, pt_vec, false); 
    }
    if(!ECALC) {
        write_data_variable(gnufilepath+"pulse_intensity.txt", n_calc, n_type, intensity, pt_vec, true); 
        write_data_variable(outfilepath+"pulse_intensity.txt", n_calc, n_type, intensity, pt_vec, false); 
    }
    gnufilepath = "outputs/gnu/sum/variable/";
    outfilepath = "outputs/out/sum/variable/";
    if(PERP_AVG) { //only averaged reuslt printed
        if(SUM and ECALC) {
            write_data_variable(gnufilepath+"photon_energy_perp.txt", n_calc, n_sum_type, wx, pt_sum_vec_perp, true);
            write_data_variable(outfilepath+"photon_energy_perp.txt", n_calc, n_sum_type, wx, pt_sum_vec_perp, false);
        }
        if(SUM and !ECALC) {
            write_data_variable(gnufilepath+"pulse_intensity_perp.txt", n_calc, n_sum_type, intensity, pt_sum_vec_perp, true); 
            write_data_variable(outfilepath+"pulse_intensity_perp.txt", n_calc, n_sum_type, intensity, pt_sum_vec_perp, false); 
        }
        gnufilepath = "outputs/gnu/variable/";
        outfilepath = "outputs/out/variable/";
        if(ECALC) {
            write_data_variable(gnufilepath+"photon_energy_perp.txt", n_calc, n_type, wx, pt_vec_perp, true); 
            write_data_variable(outfilepath+"photon_energy_perp.txt", n_calc, n_type, wx, pt_vec_perp, false); 
        }
        if(!ECALC) {
            write_data_variable(gnufilepath+"pulse_intensity_perp.txt", n_calc, n_type, intensity, pt_vec_perp, true); 
            write_data_variable(outfilepath+"pulse_intensity_perp.txt", n_calc, n_type, intensity, pt_vec_perp, false); 
        }
    }


}    

void FILEWRITER::write_data(string outfilename, int neqn, vector<vec1x > pt_vec, vector<double> norm_t_vec, bool GNUPLOT_OUT) {
    
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

void FILEWRITER::write_data_variable(string outfilename, int n_calc, int neqn, vector<double> variable, vector<vector<vec1x > > pt_vec, bool GNUPLOT_OUT) {

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

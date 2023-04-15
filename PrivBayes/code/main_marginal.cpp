#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
using namespace std;

#include "methods.h"

double exp_marginal(table& tbl, base& base, int k) {
	const vector<vector<int>> mrgs = tools::kway(k, tools::vectorize(tbl.dim));
	double err = 0.0;
	for (const vector<int>& mrg : mrgs) err += tools::TVD(tbl.getCounts(mrg), base.getCounts(mrg));
	return err / mrgs.size();
}

int main(int argc, char *argv[]) {
	// arguments
	string dataset = argv[1];
    cout << dataset <<endl;
    //age workclass fnlwgt education education-num marital occupation relationship race sex capital-gain capital-loss hours-per-week native-country salary
    double ep = stod(argv[2]);

    int rep = stoi(argv[3]);
    
    string source_dir = argv[4];
    string domain_dir = argv[5];
    string target_dir = argv[6];
    int partial = stoi(argv[7]); //if 1, then compute marginals from available partial data. else 0
    string algo_name = "PrivBayes";
    if (partial==1)
    	algo_name = algo_name + "2"; //algoname is PrivBayes2 if on missing data
	vector<double> thetas;
	for (int i = 8; i < argc; i++) {
		thetas.push_back(stod(argv[i]));
		cout << thetas.back() << "\t";
	}
	cout << endl;
	// arguments


	random_device rd;						//non-deterministic random engine
	engine eng(rd());						//deterministic engine with a random seed

	table tbl(dataset, source_dir, domain_dir, partial, true);
	printf("table reading finished \n");
	vector<int> queries = { 2, 3 };

	for (double theta : thetas) {
		printf("theta: %f \n", theta);
//		out << "theta: " << theta << endl;
		for (double epsilon : {ep}) {
		    printf("epsilon: %f \n", epsilon);
//			vector<double> err(queries.size(), 0.0);
			for (int i = 0; i < rep; i++) {
                printf("epsilon: %f, rep: %d \n", epsilon, i);
				cout << "epsilon: " << epsilon << " rep: " << i << endl;
				bayesian bayesian(eng, tbl, epsilon, theta, i, 1);
				string syn_path = target_dir + algo_name +"_run" + to_string(i) + ".syn";
                                bayesian.dump_tbl(syn_path);
			}
		}
	}
	return 0;
}

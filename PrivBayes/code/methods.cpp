#include "methods.h"
#include <cstring>
base::base(engine& eng1, table& tbl1) : eng(eng1), tbl(tbl1) {
}

base::~base() {
}



////////////////////////////// bayesian //////////////////////////////
bayesian::bayesian(engine& eng1, table& tbl1, double ep, double theta, int rep, int isGenSyn) : base(eng1, tbl1) {


//    ofstream trysyn("try.t");
	dim = tbl.dim;
	printf("building bayesian \n");
	printf("dim: %d \n", dim);
	printf("table size: %d \n", tbl1.size());

//    string value;
//    vector<int> tuple;
//    while (getline(fdata, s)) {
//        stringstream ss(s);
//        tuple.clear();
//        for (int t = 0; t < dim; t++) {
//            ss >> value;
//            tuple.push_back(translators[t]->str2int(value));
//        }
//        data.push_back(tuple);
//    }


	bound = ep * tbl.size() / (4.0 * dim * theta);		// bound to nr. of cells

	// for efficiency
	int sub = log2(bound);
	int count = 0;
	while (tbl.size() * tools::nCk(dim, sub) > 2e13) {		// default 2e9
		sub--;
		bound /= 2;
		count++;
	}
	if (count) cout << "Bound reduced for efficiency: " << count << "." << endl;
	// for efficiency
	
    if(isGenSyn == 1) {
        model = greedy(0.5 * ep);
        addnoise(0.5 * ep);
        sampling(tbl.size());

        // print_syn(tbl1, ep, rep);
    } else {
        syn = tbl;
    }
}

bayesian::~bayesian() {
}

vector<dependence> bayesian::greedy(double ep) {
	vector<dependence> model;
	double sens = tbl.sens;

	set<int> S;
	set<int> V = tools::setize(dim);
//    printf("dim: %f \n", dim);
	printf("greedy \n");
	for (int t = 0; t < dim; t++) {
	    printf("t: %d \n", t);
		vector<dependence> deps = S2V(S, V);		
		cout << deps.size() << "\t"<<endl;
		vector<double> quality;
		for (const auto& dep : deps) {
//		    printf("getScore(dep): %f \n", tbl.getScore(dep));
		    quality.push_back(tbl.getScore(dep));
		}
		dependence picked = t ? deps[noise::EM(eng, quality, ep / (dim - 1), sens)] : deps[noise::EM(eng, quality, 1000.0, sens)];
		// first selection is free: all scores are zero.

		S.insert(picked.x.first);								
		V.erase(picked.x.first);
		model.push_back(picked);
		printf("%s \n", to_string(picked).c_str());
//		cout << to_string(picked) << endl;				// debug
	}
	return model;
}

vector<dependence> bayesian::S2V(const set<int>& S, const set<int>& V) {
	vector<dependence> ans;
	for (int x : V) {
		set<vector<attribute>> exist;
		vector<vector<attribute>> parents = maximal(S, bound / tbl.getWidth(x));

		for (const vector<attribute>& p : parents)
			if (exist.find(p) == exist.end()) {
				exist.insert(p);
				ans.push_back(dependence(p, attribute(x, 0)));
			}
		if (exist.empty()) ans.push_back(dependence(vector<attribute>(), attribute(x, 0)));
	}
	return ans;
}

vector<vector<attribute>> bayesian::maximal(set<int> S, double tau) {
	vector<vector<attribute>> ans;
	if (tau < 1) return ans;
	if (S.empty()) {
		ans.push_back(vector<attribute>());
		return ans;
	}

	int last = *(--S.end());
	S.erase(--S.end());
	int depth = tbl.getDepth(last);
	set<vector<attribute>> exist;

	// with 'last' at a certain level
	for (int l = 0; l < depth; l++) {
		attribute att(last, l);
		vector<vector<attribute>> maxs = maximal(S, tau / tbl.getWidth(att));
		for (vector<attribute> z : maxs)
			if (exist.find(z) == exist.end()) {
				exist.insert(z);
				z.push_back(att);
				ans.push_back(z);
			}
	}

	// without 'last'
	vector<vector<attribute>> maxs = maximal(S, tau);
	for (vector<attribute> z : maxs)
		if (exist.find(z) == exist.end()) {
			exist.insert(z);
			ans.push_back(z);
		}

	return ans;
}

void bayesian::addnoise(double ep) {
	syn.initialize(tbl);
	for (const dependence& dep : model) {
		vector<double>& counts_syn = syn.margins[dep.cols].counts[dep.lvls];
		for (double count : tbl.getCounts(dep.cols, dep.lvls))
			counts_syn.push_back(count + noise::nextLaplace(eng, 2.0 * dim / ep));
	}
	// add consistency
}

void bayesian::sampling(int num) {
	for (int i = 0; i < num; i++) {
		vector<int> tuple(dim, 0);
		for (const dependence& dep : model) {
			vector<int> pre = tbl.generalize(
				tools::projection(tuple, dep.cols), 
				dep.cols, 
				dep.lvls);

			vector<double> conditional = syn.getConditional(dep, pre);
			tuple[dep.x.first] = noise::sample(eng, conditional);
		}
		syn.data.push_back(tuple);
	}
	syn.margins.clear();
}

string bayesian::to_string(const dependence& dep) {
	string ans = to_string(dep.x) + " <-";
	for (const auto& p : dep.p)
		ans += " " + to_string(p);
	return ans;
}

string bayesian::to_string(const attribute& att) {
	return std::to_string(att.first) + "(" + std::to_string(att.second) + ")";
}

void bayesian::printo_libsvm(const string& filename, const int& col, const set<int>& positives) {
	syn.printo_libsvm(filename, col, positives);
}

void bayesian::printo_libsvm(const string& filename, const int& col, const set<int>& positives, double prob) {
    syn.printo_libsvm(filename, col, positives, prob);
}

double bayesian::evaluate() {
	double sum = 0.0;
	for (const dependence& dep : model) sum += tbl.getMutual(dep);
	return sum;
}

// interface
vector<double> bayesian::getCounts(const vector<int>& mrg) {
	return syn.getCounts(mrg);
}

void bayesian::print_syn(table tbl, double ep, int rep) {
    ostringstream oss;
    oss << tbl.name << "-theta4-epsilon" << ep << "-rep" << rep << ".syn";
    std::ofstream syndata(oss.str());

    printf("table size: %d\n", tbl.size());
    printf("syn size: %d\n", syn.data.size());
    for(auto & i : syn.data) {
        if(i.size() == dim) {
            for(int j = 0; j < dim; j++) {
                syndata << syn.translators[j]->int2str(i[j]) << " ";
            }
        }
        syndata << endl;
    }
    try {
        printf("ready to close \n");
        syndata.close();
        printf("finished \n");
    }catch(const char* msg) {
        printf(msg);
    }
}

void bayesian::dump_tbl(string& path) {
    ofstream myfile;
    myfile.open (path);

    myfile << tbl.headerstring.substr(0, tbl.headerstring.length()-1);
    myfile << "\n";
    for (int i=0; i<syn.size(); ++i) {
        for (int j=0; j< dim; ++j) {
            myfile << syn.translators[j]->int2str(syn.data[i][j]);

            if (j < dim -1)
                myfile << ",";
            else
                myfile << "\n";
        }
    }

    myfile.close();
}
////////////////////////////	laplace //////////////////////////////
//laplace::laplace(engine& eng1, table& tbl1, double ep, const vector<vector<int>>& mrgs) : base(eng1, tbl1) {
//	double scale = 2.0 * mrgs.size() / ep;
//	for (const auto& mrg : mrgs) {
//		vector<double> counts = tbl.getCounts(mrg);
//		for (double& val : counts) val = max(0.0, val + noise::nextLaplace(eng, scale));
//		noisy[mrg] = counts;
//	}
//}
//
//laplace::~laplace() {
//}
//
//// interface
//vector<double> laplace::getCounts(const vector<int>& mrg) {
//	return noisy[mrg];
//}
//
//
//
////////////////////////////	contingency //////////////////////////////
//contingency::contingency(engine& eng1, table& tbl1, double ep) : base(eng1, tbl1) {
//	vector<int> hist = tbl.getHistogram();
//	vector<int> cells = tbl.cells(tbl.dimset());
//
//	vector<double> noisy(hist.begin(), hist.end());
//	for (double& val : noisy) val += noise::nextLaplace(eng, 2.0 / ep);
//
//	syn.copySettings(tbl);
//	vector<int> sampled = noise::sample(eng, noisy, tbl.size());
//	for (const int item : sampled)
//		syn.data().push_back(tools::decode(item, cells));
//}
//
//contingency::~contingency() {
//}
//
//// interface
//vector<double> contingency::getCounts(const vector<int>& mrg) {
//	return syn.getCounts(mrg);
//}

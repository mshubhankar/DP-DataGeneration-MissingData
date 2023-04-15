#include "table.h"
#include <iostream>
using namespace std;

table::table(const string& dataset, const string& dataset_dir, const string& domain_dir, int partial, bool func1) : func(func1) {
	// read domain
	cout<<domain_dir<<endl;
	ifstream fdomain(domain_dir);
	printf("building table \n");
    name = dataset;
	string s;
	while (getline(fdomain, s)) {
	    // cout << s <<endl;
		if (s[0] == 'D') {
//            cout << s.substr(2) << endl;
            translators.push_back(make_shared<dtranslator>(s.substr(2)));
        }
		else {
			size_t spos;
			double min = stod(s.substr(2), &spos);
			double max = stod(s.substr(2 + spos));
			translators.push_back(make_shared<ctranslator>(min, max, 4));		//internal parameter: cut into at most 2^4 blocks
		}
	}
	dim = translators.size();
	printf("dim: %d \n", dim);
	fdomain.close();


	// read data
	ifstream fdata(dataset_dir);

	printf("reading dataset \n");
	string value;
	headerstring = "";
	vector<int> tuple;
	bool header = true;
	while (getline(fdata, s)) {
		stringstream ss(s);
		int t = 0;
		tuple.clear();
		bool is_missing = false;
        string row = ss.str();
		while(getline(ss, value, ',')){

			if(header){
				headerstring.append(value);
				headerstring.append(","); //parsing header
			}
			else{
				if(value.length() != 0 && row[row.length()-1] != ','){ //Complete rows
					tuple.push_back(translators[t]->str2int(value)); //Get corresponding domain no.
                }
				else{
					if(partial==1){ //PrivBayes2 partial marginals
                        if(value.length() != 0) //When value present
                            tuple.push_back(translators[t]->str2int(value));
                        else
    						tuple.push_back(-1); //fill -1 for missing values 
						is_missing = false;
					}
					else
						is_missing = true;
				}
			}
		t+=1;
		
		}
        if (tuple.size() != dim && row[row.length()-1] == ',') //For last element in row if missing
            tuple.push_back(-1);

		if(!header && !is_missing)
			data.push_back(tuple);
		if(header)
			header=false;
	}
	fdata.close();

	double num = size();
	sens = func ? 6 / num
		: 2 / num * log2((num + 1) / 2) + (num - 1) / num * log2((num + 1) / (num - 1));

	//sens = binary ? 1 / num 
	//	: 2 / num * log2((num + 1) / 2) + (num - 1) / num * log2((num + 1) / (num - 1));
}

table::table() {
}

table::~table() {
}



int table::size() {
	return data.size();
}

int table::getDepth(int col) {		// depth setting
	return translators[col]->depth;
	// return 1;
}

vector<int> table::getDepth(const vector<int>& cols) {
	vector<int> depths;
	for (int col : cols) depths.push_back(getDepth(col));
	return depths;
}

int table::getWidth(int col, int lvl) {
	return translators[col]->size(lvl);
}

vector<int> table::getWidth(const vector<int>& cols) {
	return getWidth(cols, vector<int>(cols.size()));
}

vector<int> table::getWidth(const vector<int>& cols, const vector<int>& lvls) {
	vector<int> widths;
	for (int t = 0; t < cols.size(); t++) 
		widths.push_back(translators[cols[t]]->size(lvls[t]));
	return widths;
}

int table::getWidth(const attribute& att) {
	return translators[att.first]->size(att.second);
}

double table::getScore(const dependence& dep) {
	materialize(dep.cols, dep.lvls);
	return margins[dep.cols].scores[dep.lvls][dep.ptr];
}

double table::getMutual(const dependence& dep) {
	materialize(dep.cols, dep.lvls);
	return margins[dep.cols].mutual[dep.lvls][dep.ptr];
}

//vector<double> table::getCounts(const vector<int>& cols) {
//	materialize(cols);
//	return margins[cols].counts[vector<int>(cols.size())];
//}

vector<double> table::getCounts(const vector<int>& cols) {
	return getCounts(cols, vector<int>(cols.size()));
}

vector<double> table::getCounts(const vector<int>& cols, const vector<int>& lvls) {
	materialize(cols, lvls);
	return margins[cols].counts[lvls];
}

vector<double> table::getF(const vector<double>& counts, const vector<int>& widths) {
	vector<double> ans;
	for (int t = 0; t < widths.size(); t++) {
		vector<int> bounds(widths);
		bounds[t] = 1;	// always 0

		map<int, int> now; now[0] = 0;		// using map is faster here.
		int ceil = (size() + 1) / 2;
		vector<int> values(widths.size(), 0);
		do {
			vector<double> conditional;
			for (int x = 0; x < widths[t]; x++) {
				values[t] = x;
				conditional.push_back(counts[tools::encode(values, widths)]);
			}
			values[t] = 0;

			map<int, int> next;
			for (const auto& status : now) {
				pair<int, int> nextus(min(status.first + int(conditional[0]), ceil), min(status.second + int(conditional[1]), ceil));
				next[nextus.first] = max(next[nextus.first], status.second);
				next[status.first] = max(next[status.first], nextus.second);
			}
			now = next;
		} while (tools::inc(values, bounds));

		int best = -size();
		for (const auto& status : now)
			best = max(best, status.first + status.second - size());
		ans.push_back(double(best) / size());
	}
	return ans;
}

vector<double> table::getI(const vector<double>& counts, const vector<int>& widths) {
	vector<double> ans;
	for (int t = 0; t < widths.size(); t++) {
		vector<int> bounds(widths);
		bounds[t] = 1;	// always 0

		vector<vector<double>> joint;
		vector<int> values(widths.size(), 0);
		do {
			vector<double> conditional;
			for (int x = 0; x < widths[t]; x++) {
				values[t] = x;
				conditional.push_back(counts[tools::encode(values, widths)]);
			}
			values[t] = 0;
			joint.push_back(conditional);
		} while (tools::inc(values, bounds));

		ans.push_back(tools::mutual_info(joint));
	}
	return ans;
}

vector<double> table::getR(const vector<double>& counts, const vector<int>& widths) {
	vector<double> ans;
	for (int t = 0; t < widths.size(); t++) {
		vector<int> bounds(widths);
		bounds[t] = 1;	// always 0

		vector<vector<double>> joint;
		vector<int> values(widths.size(), 0);
		do {
			vector<double> conditional;
			for (int x = 0; x < widths[t]; x++) {
				values[t] = x;
				conditional.push_back(counts[tools::encode(values, widths)]);
			}
			values[t] = 0;
			joint.push_back(conditional);
		} while (tools::inc(values, bounds));

		ans.push_back(tools::margin_distance(joint));
	}
	return ans;
}

vector<double> table::getConditional(const dependence& dep, const vector<int>& pre) {
	vector<int> widths = getWidth(dep.cols, dep.lvls);
	int base = tools::encode(pre, widths);
	vector<int> inc(pre.size(), 0); inc[dep.ptr] = 1;
	int step = tools::encode(inc, widths);

	vector<double> counts = getCounts(dep.cols, dep.lvls);
	vector<double> conditional;
	for (int t = 0; t < widths[dep.ptr]; t++) {
		conditional.push_back(counts[base]);
		base += step;
	}
	return conditional;
}



// void table::materialize(const vector<int>& cols) {		// materialize the base of a margin
// 	if (margins.find(cols) != margins.end()) return;
// 	marginal& margin = margins[cols];
	
// 	// counts
// 	const vector<int> widths = getWidth(cols);
// 	int total = accumulate(widths.begin(), widths.end(), 1, multiplies<int>());

// 	vector<double>& counts = margin.counts[vector<int>(cols.size())];
// 	counts = vector<double>(total, 0.0);
// 	for (const auto& tuple : data) {
// 		vector<int> projected = tools::projection(tuple, cols);
// 		counts[tools::encode(projected, widths)]++;
// 	}

// 	// scores
// 	vector<double>& scores = margin.scores[vector<int>(cols.size())];
// 	scores = func ? getF(counts, widths) : getI(counts, widths);
// }
//
// void table::materialize(const vector<int>& cols, const vector<int>& lvls) {
// 	materialize(cols);
// 	marginal& margin = margins[cols];
// 	if (margin.counts.find(lvls) != margin.counts.end()) return;

// 	// counts
// 	const vector<int> widths = getWidth(cols, lvls);
// 	int total = accumulate(widths.begin(), widths.end(), 1, multiplies<int>());
// 	const vector<int> widths_0 = getWidth(cols);

// 	vector<double>& counts = margin.counts[lvls];
// 	counts = vector<double>(total, 0.0);
// 	const vector<double>& counts_0 = margin.counts[vector<int>(cols.size())];

// 	for (int code_0 = 0; code_0 < counts_0.size(); code_0++) {
// 		const vector<int> vals_0 = tools::decode(code_0, widths_0);
// 		const vector<int> vals = generalize(vals_0, cols, lvls);
// 		int code = tools::encode(vals, widths);
// 		counts[code] += counts_0[code_0];
// 	}

// 	// scores
// 	vector<double>& scores = margin.scores[lvls];
// 	scores = func ? getF(counts, widths) : getI(counts, widths);
// }
template <typename T>
bool contains_missing(vector<T> tuple, vector<T> widths){
	bool flag = false;
	for (int i = 0; i<tuple.size(); i++){
		if(tuple[i]<0 || tuple[i]>=widths[i])
			flag = true;
	}
	return flag;
}

void table::materialize(const vector<int>& cols, const vector<int>& lvls) {
	marginal& margin = margins[cols];
	if (margin.counts.find(lvls) != margin.counts.end()) return;

	// counts
	const vector<int> widths = getWidth(cols, lvls);
	int total = accumulate(widths.begin(), widths.end(), 1, multiplies<int>());
	vector<double>& counts = margin.counts[lvls];
	counts = vector<double>(total, 0.0);
	for (const auto& tuple : data) {
		vector<int> projected = generalize(tools::projection(tuple, cols), cols, lvls);
		if (!contains_missing(projected, widths))
			counts[tools::encode(projected, widths)]++;
	}
	// scores and mutual
	margin.scores[lvls] = func ? getR(counts, widths) : getI(counts, widths);
	// margin.mutual[lvls] = getI(counts, widths);
}

int table::generalize(int val, int col, int from, int to) {
	return translators[col]->generalize(val, from, to);
}

vector<int> table::generalize(const vector<int>& vals, const vector<int>& cols, const vector<int>& to) {
	vector<int> generalized;
	for (int t = 0; t < vals.size(); t++)
		generalized.push_back(generalize(vals[t], cols[t], 0, to[t]));
	return generalized;
}

vector<int> table::generalize(const vector<int>& vals, const vector<int>& cols, const vector<int>& from, const vector<int>& to) {
	vector<int> generalized;
	for (int t = 0; t < vals.size(); t++)
		generalized.push_back(generalize(vals[t], cols[t], from[t], to[t]));
	return generalized;
}


pair<int, int> table::specialize(int val, int col, int from, int to) {
	return translators[col]->specialize(val, from, to);
}

vector<pair<int, int>> table::specialize(const vector<int>& vals, const vector<int>& cols, const vector<int>& from) {
	vector<pair<int, int>> specialized;
	for (int t = 0; t < vals.size(); t++)
		specialized.push_back(specialize(vals[t], cols[t], from[t], 0));
	return specialized;
}


void table::initialize(const table& other) {
	func = other.func;
	dim = other.dim;
	translators = other.translators;
	sens = other.sens;
	data.clear();
	margins.clear();
}

void table::printo_libsvm(const string& filename, int col, const set<int>& positives) {
	ofstream fsvm(filename);
	for (const auto& tuple : data) {
		string output;
		int index = 1;

		for (int t = 0; t < dim; t++) {
			if (t == col) {
				if (positives.find(tuple[t]) != positives.end()) output = "+1" + output;
				else output = "-1" + output;
			}
			else output += " " + translators[t]->int2libsvm(tuple[t], index);
		}
		fsvm << output << endl;
	}
	fsvm.close();
}
void table::printo_libsvm(const string& filename, int col, const set<int>& positives, float prob) {
    ofstream fsvm(filename);
    for (const auto& tuple : data) {
        if(((double) rand() / (RAND_MAX)) < prob) {
            string output;
            int index = 1;

            for (int t = 0; t < dim; t++) {
                if (t == col) {
                    if (positives.find(tuple[t]) != positives.end()) output = "+1" + output;
                    else output = "-1" + output;
                } else output += " " + translators[t]->int2libsvm(tuple[t], index);
            }
            fsvm << output << endl;
        }
    }
    fsvm.close();
}
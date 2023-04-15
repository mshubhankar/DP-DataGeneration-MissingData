// #include <string>
// #include <iostream>
// #include <fstream>
// using namespace std;

// #include "methods.h"

// int main(int argc, char *argv[]) {
// 	// arguments
// 	string dataset = argv[1];
// //	string model = argv[2];
// //	cout << dataset << "-" << model << endl;

// //    double ep = stod(argv[3]);

// //	int rep = atoi(argv[2]);
//    int rep = 1;

// 	vector<double> thetas;
// 	for (int i = 2; i < argc; i++) {
// 		thetas.push_back(stod(argv[i]));
// 		cout << thetas.back() << "\t";
// 	}
// 	cout << endl;
// 	// arguments

// 	for(string model :{"all"}) {

//        int col;
//        set<int> positives = {1};

//        set<int> hospitalPositives;
//        vector<set<int>> positiveList;
//        int cols[17];

//        if (dataset == "adult") {
//            if (model == "salary") col = 14;
//            else if (model == "gender") col = 9;
//            else if (model == "education") {
//                col = 3;
//                positives = {9, 10, 11, 12, 13, 14, 15};
//            } else if (model == "marital") {
//                col = 5;
//                positives = {0};
//            } else if (model == "workclass") {
//                col = 1;
//                positives = {3, 4, 5};
//            } else if (model == "occupation") {
//                col = 6;
//                positives = {9, 10, 11, 12, 13};
//            }
//        } else if (dataset == "nltcs") {
//            if (model == "outside") col = 15;
//            else if (model == "money") col = 2;
//            else if (model == "bathing") col = 10;
//            else if (model == "traveling") col = 3;
//        } else if (dataset == "br2000") {
//            if (model == "dwell") {
//                col = 0;
//                positives = {0};
//            } else if (model == "car") {
//                col = 1;
//                positives = {0};
//            } else if (model == "child") {
//                col = 3;
//                positives = {0};
//            } else if (model == "age") {
//                col = 4;
//                positives = {0, 1, 2, 3};
//            } else if (model == "religion") {
//                col = 7;
//                positives = {0};
//            }
//        } else if (dataset == "acs") {
//            if (model == "dwell") col = 3;
//            else if (model == "mortgage") col = 4;
//            else if (model == "multigen") col = 9;
//            else if (model == "race") col = 14;
//            else if (model == "school") col = 18;
//            else if (model == "migrate") col = 20;
//        } else if (dataset == "hospital2") {
//            for (int i = 0; i < 17; i++) {
//                cols[i] = i;
//            }
//            ifstream fpicker("./cherry-picker/hospital-1");
//            printf("reading ./cherry-picker/hospital-1 \n");
//            int value;
//            string s;
//            while (getline(fpicker, s)) {
//                stringstream ss(s);
//                hospitalPositives.clear();
//                while(ss >> value){
//                    hospitalPositives.insert(value);
//                }
//                positiveList.push_back(hospitalPositives);

//            }
//            fpicker.close();
//        }

//        string name = dataset + "-" + model + "-";
//        random_device rd;                        //non-deterministic random engine
//        engine eng(rd());                        //deterministic engine with a random seed


//        ofstream out(name + ".out");
//        ofstream log(name + ".log");
//        cout.rdbuf(log.rdbuf());

//        if(dataset != "hospital2"){
//            table tbl("./data/" + dataset, true);
//            tbl.printo_libsvm(dataset + ".test", col, positives, 0.3);

//            // load testing
//            ifstream ftest(dataset + ".test");
//            string s;
//            vector<int> ytests;
//            while (getline(ftest, s)) {
//                ytests.push_back(stoi(s));
//            }
//            ftest.close();

//            for (double epsilon : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}) {
//                printf("epsilon: %f \n", epsilon);
//                string pred = name + "-epsilon" + to_string(epsilon);
//                cout << pred << endl;
//                double err = 0.0;
//                for (int i = 0; i < rep; i++) {
//                    cout << "epsilon: " << epsilon << " rep: " << i << endl;
//                    bayesian bayesian(eng, tbl, epsilon, 4.0, i, 0);
//                    bayesian.printo_libsvm(name, col, positives, 0.7);
//                    printf("%s \n", name.c_str());
//                    system(("svm-train -t 2 " + name).c_str());
//                    system(("svm-predict " + dataset + ".test " + name + ".model " + pred + ".pred").c_str());


//                    // load prediction
//                    double mismatch = 0;
//                    int ypred;
//                    ifstream fpred(pred + ".pred");
//                    for (const int &ytest : ytests) {
//                        fpred >> ypred;
//                        if (ytest != ypred) mismatch++;
//                    }
//                    fpred.close();
//                    err += mismatch / ytests.size();
//                    out << "err: " << err << endl;
//                    }
//                }
//            cout << endl;
//        } else {
//            table tbl1("./data/hospital2-theta4-epsilon1-rep0", true);
//            table tbl2("./data/hospital2-theta4-epsilon10-rep0", true);
//            table ori("./data/hospital2", true);

//            for (int col: cols) {
//                printf("col: %d \n", col);
//                set<int> positive = positiveList.at(col);
//                // Debug
//                if (col == 0) {
//                    auto pos = positive.find(5919);
//                    printf("The set elements after 5919 are: ");
//                    for (auto it = pos; it != positive.end(); it++)
//                        printf("%d ", *it);
//                    printf("\n");
//                }

//                string test = dataset + "-"+ to_string(col) + ".test";
//                ori.printo_libsvm(test, col, positive, 0.3);
// //                tbl1.printo_libsvm(tbl1.name + ".test", col, positive, 0.3);
// //                tbl2.printo_libsvm(tbl2.name + ".test", col, positive, 0.3);

//                // load testing
//                ifstream ftest(test);
//                string s;
//                vector<int> ytests;
//                while (getline(ftest, s)) {
//                    ytests.push_back(stoi(s));
//                }
//                ftest.close();

//                double err = 0.0;
//                for (int i = 0; i < rep; i++) {
//                    printf("rep: %d \n", i);
//                    for (table tbl: {ori}) {
//                        float epsilon;
//                        if (tbl.name == "./data/hospital2-theta4-epsilon1-rep0") {
//                            epsilon = 1;
//                        } else if (tbl.name == "./data/hospital2-theta4-epsilon10-rep0") {
//                            epsilon = 10;
//                        } else {
//                            epsilon = 0;
//                        }
//                        printf("epsilon: %f \n", epsilon);
//                        string name = dataset + "-" + model + "-col-" + to_string(col);
//                        string pred = name + "-epsilon" + to_string(int(epsilon));
//                        printf("pred: %s \n", pred.c_str());

//                        cout << pred << endl;
//                        cout << "epsilon: " << epsilon << " rep: " << i << endl;
//                        bayesian bayesian(eng, tbl, epsilon, 4.0, i, 0);
//                        bayesian.printo_libsvm(pred, col, positive, 0.7);
//                        printf("pred: %s \n", pred.c_str());
//                        system(("svm-train -t 2 " + pred).c_str());
//                        printf("trained\n");
//                        printf("dataset.test: %s \n", test.c_str());
//                        system(("svm-predict " + test + " " + pred + ".model " + pred + ".pred").c_str());

//                        // load prediction
//                        double mismatch = 0;
//                        int ypred;
//                        ifstream fpred(pred + ".pred");
//                        for (const int &ytest : ytests) {
//                            fpred >> ypred;
//                            if (ytest != ypred) mismatch++;
//                        }
//                        fpred.close();
//                        err += mismatch / ytests.size();
//                        printf("err: %f\n", err);
//                        out << "err: " << err << endl;

//                    }
//                }
//                cout << endl;
//            }
//        }


//        out.close();
//        log.close();



// //        for (double theta : thetas) {
// //            cout << "theta: " << theta << endl;
// //            for (double epsilon : {0.05, 0.1, 0.2, 0.4, 0.8, 1.6}) {
// //                printf("epsilon: %f \n", epsilon);
// //                string pred = name + "-epsilon" + to_string(epsilon) + "-theta" + to_string(theta);
// //                cout << pred << endl;
// //                double err = 0.0;
// //                for (int i = 0; i < rep; i++) {
// //                    if (dataset != "hospital") {
// //                        cout << "epsilon: " << epsilon << " rep: " << i << endl;
// //                        bayesian bayesian(eng, tbl, epsilon, theta, i, 0);
// //                        bayesian.printo_libsvm(name, col, positives, 0.7);
// //                        printf("%s \n", name.c_str());
// //                        system(("svm-train -t 2 " + name).c_str());
// ////                    printf("trained");
// //                        system(("svm-predict " + dataset + ".test " + name + ".model " + pred + ".pred").c_str());
// //
// //                        // load prediction
// //                        double mismatch = 0;
// //                        int ypred;
// //                        ifstream fpred(pred + ".pred");
// //                        for (const int &ytest : ytests) {
// //                            fpred >> ypred;
// //                            if (ytest != ypred) mismatch++;
// //                        }
// //                        fpred.close();
// //                        err += mismatch / ytests.size();
// //                        out << "err: " << err << endl;
// //                    } else {
// //                        for (table tbl: {tbl1, tbl2}) {
// //
// //                        }
// //                    }
// //                }
// //
// //            }
// //            cout << endl;
// //        }

//    }
// 	return 0;
// }
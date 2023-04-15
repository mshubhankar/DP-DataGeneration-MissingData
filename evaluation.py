import os
import re
import concurrent.futures
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import combinations
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,\
                             RandomForestClassifier, BaggingClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler


regex = re.compile(r"\[|\]|<", re.IGNORECASE)
random_state = 42
sample_size = 5000
test_size = 0.1

models = [LogisticRegression(random_state=random_state),
          AdaBoostClassifier(random_state=random_state),
          GradientBoostingClassifier(random_state=random_state),
          xgb.XGBClassifier(random_state=random_state),
          RandomForestClassifier(random_state=random_state),
          BernoulliNB(),
          DecisionTreeClassifier(random_state=random_state),
          BaggingClassifier(random_state=random_state),
          MLPClassifier(random_state=random_state),
          ]

def _get_targets(path):
    """
    Return target and postive labels for ML models
    """
    target_attrs = []
    pos_values = []

    if 'adult' in path:
        target_attrs = ['income', 'sex', 'marital-status', 'education',
                        'age', 'workclass', 'education-num',
                        'occupation', 'relationship', 'race',
                        'capital-gain', 'capital-loss',
                        'hours-per-week', 'native-country'
                        ]
        pos_values = [['>50k'], ['female'], ['never-married'],
                      ['some-college', 'assoc-voc', 'assoc-acdm', 'masters', 'doctorate', \
                        'bachelors', 'prof-school'],
                      [50, 100],
                      ['state-gov', 'federal-gov', 'local-gov'],
                      [10, 17],
                      ['prof-specialty'],
                      ['unmarried'],
                      ['black'],
                      [0, 1000], [0, 1000],
                      [40, 100],
                      ['united-states']
                      ]
    if 'bank' in path:
        target_attrs = ['age','job','marital','education','default','balance','housing', \
                        'loan','contact','day','month','duration','campaign','pdays', \
                        'previous','poutcome','y'
                        ]
        pos_values = [[18,50], ['blue-collar', 'management' , 'admin'],
                        ['married'], ['primary'], ['yes'], [15000, 102127], ['yes'], 
                        ['yes'], ['telephone'], [1, 15], ['jan', 'feb', 'mar', 'apr', 'may', 'june'], 
                        [500, 4918], [1, 30], [400, 870], [0, 100], ['success'], ['yes']
                      ]

    elif 'tpch' in path:
        target_attrs = ['c_custkey','c_nationkey','c_acctbal','c_mktsegment',
                        'n_name','n_regionkey','o_orderpriority','o_totalprice','o_orderstatus']
        pos_values = [  # total = 19951
            [f'custkey_{code}' for code in range(0, 110000)],
            [f'nationkey_{code}' for code in range(0, 12)],
            [-1000,8000],  
            ['automobile', 'household', 'furniture', 'building'], 
            ['vietnam','china','russia','jordan','indonesia','japan','india','iran','saudiarabia'\
                ,'iraq'],
            ['regionkey_2', 'regionkey_0', 'regionkey_3'],  
            ['2-high','3-medium', '5-low', '4-notspecified'],  
            [0,300000],
            ['o'],
        ]
    return target_attrs, pos_values

def _prep_dfs(path_true, path_syn, bin_num, clip_num, scale_num):
    """
    Take path to true and synthetic data, and parse into dataframe:
        - for categorical attributes, format to lower case
        - for numeric attribute, clip into the active domain based on the true dataset, bin into 16
    Return two dataframes, and a list of attributes
    """
    df_true = pd.read_csv(path_true)
    df_syn = pd.read_csv(path_syn)

    for df in [df_true, df_syn]:
        df.dropna(axis=0, inplace=True)

    attrs = df_true.columns.tolist()
    assert set(attrs) == set(df_syn.columns.tolist())
    for attr in attrs:
        df_syn[attr] = df_syn[attr].astype(df_true[attr].dtype)
    all_num_attrs = df_true.select_dtypes(include=np.number).columns.tolist()
    maxs = {}
    mins = {}

    n_bin = 16

    for attr in all_num_attrs:
        maxs[attr] = max(df_true[attr])
        mins[attr] = min(df_true[attr])

    for df in [df_true, df_syn]:
        for attr in attrs:
            if attr not in all_num_attrs:
                df[attr] = df[attr].str.lower()
            else:
                if clip_num:
                    df[attr].clip(lower=mins[attr], upper=maxs[attr], inplace=True)
                if bin_num:
                    df[attr].clip(lower=mins[attr], upper=maxs[attr], inplace=True)
                    bin_width = (maxs[attr] - mins[attr]) / n_bin

                    values = np.floor((df[attr] - mins[attr]) / bin_width)
                    df[attr] = values
                    df[attr] = df[attr].astype(int)

                if scale_num:
                    scaler = MinMaxScaler()
                    num_attrs = df.select_dtypes(include=np.number).columns.tolist()
                    df[num_attrs] = scaler.fit_transform(df[num_attrs])

    return df_true, df_syn, attrs

def _compute_margianl_diff(df_true_o, df_syn_0, combo_attrs):
    df_true = df_true_o.copy()
    df_syn = df_syn_0.copy()

    for df in [df_true, df_syn]:
        for attr in combo_attrs:
            df[attr] = df[attr].map(str)

    true_combo = df_true[combo_attrs].agg('_'.join, axis=1)
    syn_combo = df_syn[combo_attrs].agg('_'.join, axis=1)

    true_v_counts = true_combo.value_counts()
    syn_v_counts = syn_combo.value_counts()

    list_dict = [dict(), dict()]

    allkey = set()

    for df_idx, v_counts in enumerate([true_v_counts, syn_v_counts]):
        crnt_dict = list_dict[df_idx]

        keys = v_counts.index.tolist()

        for key in keys:
            count = v_counts[key]
            crnt_dict[key] = count
            allkey.add(key)

    diff = 0
    # print(combo_attrs)
    for key in sorted(allkey):
        count0 = list_dict[0].get(key, 0)
        count1 = list_dict[1].get(key, 0)
        # print(key, count0/ len(df_true), count1/ len(df_syn))
        diff += abs(count0 / len(df_true) - count1 / len(df_syn))

    return diff



def validate_kway_marginal(k, path_true, path_syn):
    """
    Compute the k-way marginal difference between the true and synthetic data
    :param k dimension of the marginals
    :param path_true path to the true data
    :param path_syn path to the synthetic data
    """

    df_true, df_syn, attrs = _prep_dfs(path_true, path_syn, True, True, False)

    combs = combinations(range(len(attrs)), k)
    combos_attrs = []
    for combo in combs:
        combo_attrs = [attrs[x] for x in combo]
        combos_attrs.append(combo_attrs)

    sum_diff = 0
    count = 0
    msg = f'EXP_{k}WAY\n{path_syn}\nattributes\t\t {k}-way marginals\n'
    # combos_attrs = [['c_custkey']]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4)) as executor:

        futures = [executor.submit(_compute_margianl_diff, df_true, df_syn, combo_attrs) \
                                                    for combo_attrs in combos_attrs]

        diffs = [f.result() for f in futures]

        for idx, diff in enumerate(diffs):
            combo_attrs = combos_attrs[idx]
            sum_diff += diff
            count += 1
            msg += f"{combo_attrs}\t\t {'{:.7f}'.format(diff / 2)}\n"

    average = sum_diff / 2 / count
    msg += f"average= {'{:.7f}'.format(average)}"
    print(msg)

    return msg+'\n'

def _validate_accuracy_model(model, syn_X_train, syn_y_train,
                             validate_on_true, syn_X_test, syn_y_test, true_X_test, true_y_test):

    single_label = False
    if len(syn_y_train.unique()) < 2:
        # no need to train this model, do voting on majority
        single_label = True
        label = syn_y_train.unique()[0]
    else:
        clf_syn = model.fit(syn_X_train, syn_y_train)

    if validate_on_true:
        # syn is true
        if single_label:
            syn_y_pred = [label] * len(syn_X_test)
            syn_y_pred_proba = [label] * len(syn_X_test)
        else:
            syn_y_pred = clf_syn.predict(syn_X_test)
            syn_y_pred_proba = clf_syn.predict_proba(syn_X_test)[:, 1]

        syn_acc = metrics.accuracy_score(syn_y_test, syn_y_pred, normalize=True)
        syn_pre = metrics.precision_score(syn_y_test, syn_y_pred)
        syn_recall = metrics.recall_score(syn_y_test, syn_y_pred)
        syn_f1 = metrics.f1_score(syn_y_test, syn_y_pred)
        try:
            syn_auroc = metrics.roc_auc_score(syn_y_test, syn_y_pred_proba)
        except ValueError:
            syn_auroc = 0

        syn_auprc = metrics.average_precision_score(syn_y_test, syn_y_pred_proba)

    else:

        if single_label:
            syn_y_pred = [label] * len(true_X_test)
            syn_y_pred_proba = [label] * len(true_X_test)
        else:
            syn_y_pred = clf_syn.predict(true_X_test)
            syn_y_pred_proba = clf_syn.predict_proba(true_X_test)[:, 1]

        syn_acc = metrics.accuracy_score(true_y_test, syn_y_pred, normalize=True)
        syn_pre = metrics.precision_score(true_y_test, syn_y_pred)
        syn_recall = metrics.recall_score(true_y_test, syn_y_pred)
        syn_f1 = metrics.f1_score(true_y_test, syn_y_pred)
        try:
            syn_auroc = metrics.roc_auc_score(true_y_test, syn_y_pred_proba)
        except ValueError:
            syn_auroc = 0
        syn_auprc = metrics.average_precision_score(true_y_test, syn_y_pred_proba)

    return [syn_acc, syn_pre, syn_recall, syn_f1, syn_auroc, syn_auprc]

def validate_accuracy(path_true, path_syn):
    """
    Validation function to test model accuracy and F1 for all models
    :param path_true path to the true data
    :param path_syn path to the synthetic data
    """

    validate_on_true = False
    if path_syn == path_true:
        validate_on_true = True

    df_true, df_syn, attrs = _prep_dfs(path_true, path_syn, bin_num=False, clip_num=False,\
                                         scale_num=False)

    target_attrs, pos_values = _get_targets(path_true)
    num_attrs = list(df_true.select_dtypes(include=np.number).columns)

    true_len = df_true.shape[0]
    syn_len = df_syn.shape[0]
    sn = min(sample_size, syn_len)

    df_true = df_true.sample(n=sn, random_state=random_state)
    df_syn = df_syn.sample(n=sn, random_state=random_state)

    df_true_size = len(df_true)
    df_syn_size = len(df_syn)

    msg = f'EXP_ACCURACY\n{path_syn}\n'
    for attr_idx, target_attr in enumerate(target_attrs):
        df = df_true.append(df_syn).copy()

        crnt_num_attrs = num_attrs.copy()
        if target_attr in crnt_num_attrs:
            crnt_num_attrs.remove(target_attr)
        scaler = MinMaxScaler()
        df[crnt_num_attrs] = scaler.fit_transform(df[crnt_num_attrs])

        if target_attr in num_attrs:
            ranges = pos_values[attr_idx]
            left, right = ranges[0], ranges[1]
            df[target_attr] = [1 if left <= value < right else 0 for value in df[target_attr]]
        else:
            df[target_attr] = [1 if value in pos_values[attr_idx] else 0\
                                         for value in df[target_attr]]

        df_dummies = pd.get_dummies(df)

        df_dummies.columns = [regex.sub("_", col) if any(x in str(col)\
                                     for x in set(('[', ']', '<'))) else col
                              for col in df_dummies.columns.values]

        dummies_true = df_dummies.head(df_true_size)
        dummies_syn = df_dummies.tail(df_syn_size)

        dummies_attrs = df_dummies.columns.tolist()

        dummies_true_X = dummies_true[[v for v in dummies_attrs if v != target_attr]]
        dummies_true_y = dummies_true[target_attr]
        dummies_syn_X = dummies_syn[[v for v in dummies_attrs if v != target_attr]]
        dummies_syn_y = dummies_syn[target_attr]

        true_X_train, true_X_test, true_y_train, true_y_test = \
            train_test_split(dummies_true_X, dummies_true_y, random_state=random_state,\
                            test_size=test_size)
        syn_X_train, syn_X_test, syn_y_train, syn_y_test = \
            train_test_split(dummies_syn_X, dummies_syn_y, random_state=random_state,\
                             test_size=test_size)

        syn_accs = []
        syn_pres = []
        syn_recalls = []
        syn_f1s = []
        syn_aurocs = []
        syn_auprcs = []
        msg += f'\nclassifier\t\t\ttarget_attr\tsyn_acc\tsyn_pre\tsyn_recall\tsyn_f1\tauroc\tauprc\n'

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, os.cpu_count() + 4))\
                                                     as executor:
            futures = [executor.submit(_validate_accuracy_model, model, syn_X_train, syn_y_train,\
                     validate_on_true, syn_X_test, syn_y_test, true_X_test, true_y_test)\
                     for model in models]

            diffs = [f.result() for f in futures]

            for idx, diff in enumerate(diffs):
                model = models[idx]
                syn_acc = diff[0]
                syn_pre = diff[1]
                syn_recall = diff[2]
                syn_f1 = diff[3]
                syn_auroc = diff[4]
                syn_auprc = diff[5]
                msg += f"{model.__class__.__name__}\t{target_attr}\t" \
                       f"{'{:.2f}'.format(syn_acc)}\t" \
                       f"{'{:.2f}'.format(syn_pre)}\t" \
                       f"{'{:.2f}'.format(syn_recall)}\t" \
                       f"{'{:.2f}'.format(syn_f1)}\t" \
                       f"{'{:.2f}'.format(syn_auroc)}\t" \
                       f"{'{:.2f}'.format(syn_auprc)}\n"
                syn_accs.append(syn_acc)
                syn_pres.append(syn_pre)
                syn_recalls.append(syn_recall)
                syn_f1s.append(syn_f1)
                syn_aurocs.append(syn_auroc)
                syn_auprcs.append(syn_auprc)

        metrics_size = len(models)
        msg += f"average\t{target_attr}\t" \
               f"{'{:.2f}'.format(sum(syn_accs)/metrics_size)}\t" \
               f"{'{:.2f}'.format(sum(syn_pres) / metrics_size)}\t" \
               f"{'{:.2f}'.format(sum(syn_recalls) / metrics_size)}\t" \
               f"{'{:.2f}'.format(sum(syn_f1s) / metrics_size)}\t" \
               f"{'{:.2f}'.format(sum(syn_aurocs)/metrics_size)}\t" \
               f"{'{:.2f}'.format(sum(syn_auprcs)/metrics_size)}\n"

    print(msg)
    return msg+'\n'
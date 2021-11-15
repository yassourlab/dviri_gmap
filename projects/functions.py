from collections import defaultdict

import numpy as np
import pandas as pd
# from projects.consts import META_PATH
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix,plot_confusion_matrix
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict
from plotly.offline import download_plotlyjs, plot,iplot

norm_l7_path = "/mnt/c/Users/dviri/Google Drive/University/Masters/MoranLab/Dvir/GMAT/projects/py/nb/data/feature-table-norm-l7.csv"
l7_path = "/mnt/c/Users/dviri/Google Drive/University/Masters/MoranLab/Dvir/GMAT/projects/py/nb/data/feature-table-l7.csv"
norm_l6_path = "/mnt/c/Users/dviri/Google Drive/University/Masters/MoranLab/Dvir/GMAT/projects/py/nb/data/feature-table-norm-l6.csv"
l6_path = "/mnt/c/Users/dviri/Google Drive/University/Masters/MoranLab/Dvir/GMAT/projects/py/nb/data/feature-table-l6.csv"


def build_trainig_data(merge_df, meta_idx, test_size=0.25, split_by_control=False):
    if split_by_control:
        # Don't know where na symtpoms should go. Drop them
        na_symptoms = merge_df[pd.isna(merge_df.symptoms)]
        merge_df = merge_df.drop(na_symptoms.index)
        control_data = merge_df[merge_df.symptoms == "Control"]
        ap_data = merge_df.drop(control_data.index)  # Everyone that is not control group

        X_train = ap_data.iloc[:, :meta_idx].values
        X_test = control_data.iloc[:, :meta_idx].values

        meta_train = ap_data.iloc[:, meta_idx:]
        y_train = meta_train.loc[:, 'visit_age_mo'].values

        meta_test = control_data.iloc[:, meta_idx:]
        y_test = meta_test.loc[:, 'visit_age_mo'].values

        control_data = control_data.groupby(['record_id']).apply(lambda x: split_is_train(x, 1 - test_size))
        X_control_train = control_data.loc[control_data.is_train].iloc[:, :meta_idx]
        y_control_train = control_data.loc[control_data.is_train].iloc[:, meta_idx:].loc[:, 'visit_age_mo'].values

        X_control_test = control_data.drop(X_control_train.index).iloc[:, :meta_idx]
        y_control_test = control_data.drop(X_control_train.index).iloc[:, meta_idx:].loc[:, 'visit_age_mo'].values

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, 'y_test': y_test,
                "X_control_train": X_control_train,
                "y_control_train": y_control_train,
                "X_control_test": X_control_test,
                "y_control_test": y_control_test,
                "control_data": control_data,
                'ap_data': ap_data
                }
    else:
        ap_data = merge_df
        X = merge_df.iloc[:, :meta_idx].values
        meta = merge_df.iloc[:, meta_idx:]
        y = meta.loc[:, 'visit_age_mo'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=666)

        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, 'y_test': y_test}


def quantaize_age_mo(metadata: pd.DataFrame, r_factor: int = 5, print_hist=False, inplace=False) -> pd.DataFrame:
    """ Given a rounding factor, will round the decimal point of the age in month to the decimal rounoding of that number.
    for example for r_factor 5 (default) and numbers 0.1,0.2,0.3,0.4,0.5,0.8 will return the list of 0.0,0.5,0.5,0.5,0.5,1.0,
    if print_hist is True, will print histogram of new df visit_age_mo
    """

    meta_copy = metadata.copy() if not inplace else metadata
    meta_copy.visit_age_mo = ((meta_copy.visit_age_mo / r_factor).round(1) * r_factor).round(1)
    if print_hist:
        unique_visit_mo = meta_copy.groupby('visit_age_mo')['sampleID'].nunique()
        print(unique_visit_mo)
        unique_visit_mo.hist()

    return meta_copy


def evaluate(model, features, labels, err_margin=0.3, verbose=True):
    predictions = model.predict(features)
    loss = abs(predictions - labels)
    succ = np.count_nonzero(loss < err_margin) / len(loss)

    if verbose:
        print('Model Performance')
        print('Average Error: {:0.4f} .'.format(np.mean(loss)))
        print('Accuracy = {:0.2f}%.'.format(succ))

    return succ


def get_random_search_parameters():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=2000, num=50)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = np.linspace(5, 100, num=5, dtype=int).tolist() + [None]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 20, 40]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6, 10,40]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap,
                   'random_state': [666]}
    return random_grid


def get_linear_regressor(merged_training_df, test_map, pred_name, return_data=False):
    #     test_map = merged_training_df.tt == 'test'
    training_data = merged_training_df.iloc[:, :meta_idx].loc[test_map]
    assert len(training_data) > 0, "test map didn't return any data point"
    predict = merged_training_df[pred_name]
    #     predict = best_model.predict(merged_training_df.iloc[:,:meta_idx].loc[test_map].values)
    label = merged_training_df[test_map].visit_age_mo.values
    # X = np.stack([label,predict]).T
    X = label.reshape(-1, 1)
    # sort_idxs= np.argsort(label)
    y = predict[test_map]
    # Create linear regression object

    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    if return_data:
        return regr, X

    return regr


def draw_regresssion(merged_training_df, test_map, fig, name='regression', pred_name='predict', color=None):
    regr, X = get_linear_regressor(merged_training_df, test_map, pred_name, return_data=True)
    reg_predict = regr.predict(X)
    fig.add_trace((go.Scatter(x=X[:, 0], y=reg_predict,
                              mode='lines',
                              name=name,
                              marker={"color": color},
                              line=dict(width=3))))
    return fig


def draw_xy_line(fig, max_x=14, line_width=3):
    """
    Draw x=y line on figure frin x=0 to x=max_x
    """
    x = list(np.arange(0, max_x))
    fig.add_trace((go.Scatter(x=x, y=x,
                              mode='lines',
                              name='label',
                              line=dict(width=line_width))))
    return fig


def calc_regressor_control(merge_df, params, n_samples=20, err_margin=1.0,
                           print_accuracy=True, split_control=True, train_ratio=0.8):
    if split_control:
        loss_lst = [list(), list(), list()]
        acc_lst = [list(), list(), list()]
    else:
        loss_lst = [list(), list()]
        acc_lst = [list(), list()]

    best_features = list()
    for i in tqdm(range(n_samples)):
        data = get_tt_data(merge_df, split_control=split_control, train_ratio=train_ratio)
        #     control_train_df,control_test_df,test_df = get_tt_data()
        X_train = data[0].iloc[:, :meta_idx].values
        y_train = data[0].visit_age_mo
        best_model = RandomForestRegressor(**params)
        best_model.fit(X_train, y_train)

        for i in range(len(acc_lst)):
            X = data[i].iloc[:, :meta_idx].values
            y = data[i].visit_age_mo
            predict = best_model.predict(X)

            label = y
            loss_arr = abs(predict - label)
            succ = np.count_nonzero(loss_arr < err_margin) / len(loss_arr)
            loss = np.mean(loss_arr)

            loss_lst[i].append(loss)
            acc_lst[i].append(succ)

        importances = best_model.feature_importances_
        regress_indices = np.argsort(importances)[::-1]
        best_features.append(regress_indices)

    if print_accuracy:
        regr_print_avg_acc(loss_lst, acc_lst, split_control)

    return loss_lst, acc_lst, best_features


def regr_print_avg_acc(loss_lst, acc_lst, split_control):
    if len(loss_lst) == 3:
        df_enum = ["control_train_df", "control_test_df", "test_df"]
    elif len(loss_lst) == 2:
        df_enum = ["train_df", "test_df"]
    for i in range(len(loss_lst)):
        print(df_enum[i])
        print(f"AVG loss = {(np.sum(loss_lst[i]) / len(acc_lst[i])).round(2)}")
        print(f"AVG acc = {(np.sum(acc_lst[i]) / len(acc_lst[i])).round(2)}")
        print(f"MIN acc = {np.min(acc_lst[i]).round(2)}")
        print(f"MAX acc = {np.max(acc_lst[i]).round(2)}")
        print()


def prepare_features_pos_score(best_features):
    features_scores_d = defaultdict(lambda: 0)
    for arr in best_features:
        for i, num in enumerate(arr):
            features_scores_d[num] += i
    return features_scores_d


def get_gmap_data(data_path: str):
    """
    return merge_df,meta_idx
    """
    otu_data = pd.read_csv(data_path, sep='\t', index_col=['OTU ID'])
    examples_data = otu_data.T
    metadata = pd.read_csv(META_PATH, sep='\t')
    merge_df = examples_data.merge(metadata, left_index=True, right_on=['sampleID'])
    merge_df = merge_df.assign(is_control=(merge_df.symptoms == 'Control'))
    meta_idx = examples_data.shape[1]
    return merge_df, meta_idx


def split_train_test(x, split_control=True, train_ratio=0.75):
    #     import ipdb;ipdb.set_trace()
    rand_val = np.random.rand()
    is_train = rand_val < train_ratio
    if not split_control:
        tt = 'train' if is_train else 'test'

    else:
        if not any(x.is_control):
            tt = 'test'
        else:
            tt = 'control_train' if is_train else 'control_test'

    x = x.assign(tt=tt)
    return x


def get_merged_tt_df(merge_df, split_control=True, train_ratio=0.85, inplace=False):
    """
    Add tt division for the given dataframe with ratio. group by record_id to make sure same baby won't be both in train and test.
    """

    tt_df = merge_df.groupby(['is_control', 'record_id']).apply(
        lambda x: split_train_test(x, split_control, train_ratio)).reset_index(drop=True)

    if inplace:

        merge_df['tt'] = tt_df.tt.reset_index(drop=True)
    else:
        return tt_df


def get_tt_data(merge_df, split_control=True, train_ratio=0.85):
    """
    Return a tuple of control_train_df,control_test_df,test_df
    """
    merged_training_df = get_merged_tt_df(merge_df, split_control, train_ratio)
    test_df = merged_training_df[merged_training_df.tt == 'test']
    if split_control:
        control_train_df = merged_training_df[merged_training_df.tt == 'control_train']
        control_test_df = merged_training_df[merged_training_df.tt == 'control_test']
        return control_train_df, control_test_df, test_df
    else:
        train_df = merged_training_df[merged_training_df.tt == 'train']
        return train_df, test_df


def get_regression_predict(merge_df, best_regress_params, split_control=True, train_ratio=0.85):
    """ Train and predict on a regression model with the regress_params.
    Add the columns predict,label,loss to the dataframe and return it.
    """
    regression_model = RandomForestRegressor(**best_regress_params)

    data = get_tt_data(merge_df, split_control=split_control, train_ratio=0.85)
    X = data[0].iloc[:, :meta_idx].values
    y = data[0].visit_age_mo
    regression_model.fit(X, y)

    merged_training_df = pd.concat(data)
    cols = merged_training_df.columns.tolist()

    predict = regression_model.predict(merged_training_df.iloc[:, :meta_idx].values)
    label = merged_training_df.visit_age_mo
    loss_arr = abs(predict - label)
    merged_training_df = merged_training_df.assign(predict=predict, label=label, loss=loss_arr)
    return merged_training_df




###################### CLASSIFICATION
def get_avg_accuracy(clf, merge_df, meta_idx, num_runs, y=None, train_ratio=0.8, copy_df=False, verbose=True):
    """
    Returns:
        avg_train_acc, avg_test_acc
    """
    if copy_df:
        merge_df = merge_df.copy()

    train_acc_arr = list()
    test_acc_arr = list()
    y = 'sample_time' if y is None else y
    for_func = tqdm(range(num_runs)) if verbose else range(num_runs)
    for i in for_func:
        train_acc, test_acc = get_classifier_accuracy(clf, merge_df, meta_idx, y=y, train_ratio=train_ratio)

        train_acc_arr.append(train_acc)
        test_acc_arr.append(test_acc)

    avg_train_acc = np.average(train_acc_arr)
    avg_test_acc = np.average(test_acc_arr)

    if verbose:
        print(f"Train AVG accuracy {avg_train_acc}")
        print(f"Test AVG accuracy {avg_test_acc}")

    return avg_train_acc, avg_test_acc


def get_classifier_accuracy(clf, merge_df_tt, meta_idx, y=None, train_ratio=0.8, split_tt=True, return_all=False):
    """get Classification classifier accuracy by training on the given dataframe.
    Args:
        split_tt: If True, will give the merge_df_tt a new tt assignment. Otherwise, assume that the DataFrame already
        has tt and uses it.
        return_all: If true, will return a dictionary with "test_pred", "train_pred", "merge_df_tt"
    Returns:
        train_acc,test_acc
    """
    y = 'sample_time' if y is None else y
    if split_tt:
        get_merged_tt_df(merge_df_tt, split_control=False, train_ratio=train_ratio, inplace=True)
    else:
        assert 'tt' in merge_df_tt, "Must pass a df with 'tt' division if split_tt is False "

    train_df = merge_df_tt[merge_df_tt.tt == 'train']
    test_df = merge_df_tt[merge_df_tt.tt == 'test']

    X_train = train_df.iloc[:, :meta_idx].values
    y_train = train_df[y].values

    X_test = test_df.iloc[:, :meta_idx].values
    y_test = test_df[y].values

    #     clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    train_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, pred)

    if return_all:
        res_dict = {
            "test_pred": pred,
            "train_pred": train_pred,
            "merge_df_tt": merge_df_tt,
            "train_df": train_df,
            "test_df": test_df
        }
        return res_dict

    return train_acc, test_acc


def parameter_search_classifier(X_train, y_train):
    random_grid = get_random_search_parameters()
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3, verbose=3,
                                   random_state=666, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)

    best_params = rf_random.best_params_
    return best_params


def get_confusion_matrix(test_df, pred, classes_arr, return_count=False, label_col='kmeans_label', normalize='index'):
    """
    test_df: DataFrame in the shape of (N,c) where N is the number of samples the prediciton was made on. Needs to have columns label_col
    pred: An array of a classifier prediction over which the confusion matrix will be. Need to be in shape (N,)
    classes_arr: An Array with the classes in the classification. The order the classes will be, is the order in the output table
    normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
    Normalize by dividing all values by the sum of values.
    """
    #     if classes_arr is None:
    #         classes_arr = ['onemonth', 'twomonth','fourmonth', 'sixmonth', 'ninemonth' , 'oneyear']

    df_mapping = pd.DataFrame({'sample_time': classes_arr})
    sort_mapping = df_mapping.reset_index().set_index('sample_time')

    pred_merge_df = test_df.copy()
    pred_merge_df['pred_names'] = pred
    pred_merge_df['sample_time_num'] = pred_merge_df[label_col].map(sort_mapping['index'])
    #     pred_merge_df['kmeans_label'].unique(),pred_merge_df['sample_time_num'].unique()
    pred_merge_df['pred_names_num'] = pred_merge_df['pred_names'].map(sort_mapping['index'])

    pred_s = pd.Series(pred_merge_df.sample_time_num, name='pred')
    label_s = pd.Series(pred_merge_df.pred_names_num, name='label')

    ct = pd.crosstab(label_s, pred_s, normalize=normalize)
    mapping_dict = df_mapping.to_dict()['sample_time']
    ct = ct.rename(columns=mapping_dict, index=mapping_dict)

    if not return_count:
        return ct

    cnt = pd.crosstab(label_s, pred_s)
    cnt = cnt.rename(columns=mapping_dict, index=mapping_dict)

    mapping_dict = df_mapping.to_dict()['sample_time']
    ct = ct.rename(columns=mapping_dict, index=mapping_dict)
    return ct, cnt


def get_classification_random_function(merged_training_df, label_name='sample_time', n_bins=50):
    X = merged_training_df.iloc[:, :meta_idx].values
    counts = merged_training_df[label_name].value_counts()
    values = counts.values
    bins = counts.index
    #     values, bins = np.histogram(label, bins=len(np.unique(label))-1)
    prob = values / np.sum(values)
    #     import ipdb;ipdb.set_trace()
    return lambda size: np.random.choice(bins, size=size, p=prob)


def cat_to_bin(classes_arr):
    return lambda x: 'onemonth' if classes_arr.index(x) < 3 else 'oneyear'


def get_clustered_classification(merge_df, n_clusters, labels=None, remove_sick=True):
    assert labels is None or len(labels) >= n_clusters, "Labels doesn't match the number of wanted clusters"

    if remove_sick:
        merge_df = merge_df[merge_df.sample_time != 'sick']

    X = merge_df.visit_age_mo.to_numpy().reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=666).fit(X)
    args_res = np.argsort(kmeans.cluster_centers_.reshape(-1))

    if labels is not None:
        labels = list(map(lambda x: labels[np.where(args_res == x)[0][0]], kmeans.labels_))
    else:
        labels = kmeans.labels_

    merge_df = merge_df.assign(kmeans_label=labels, kmean_num_label=kmeans.labels_)
    return merge_df


def get_avg_classification_random_accuracy(merge_df, n_runs: int, label_name='kmean_num_label', split_control=False,
                                           train_ratio=0.8, verbose=True):
    acc_arr = np.empty((n_runs, 2))
    for i in tqdm(range(n_runs)):
        acc_arr[i] = get_classification_random_accuracy(merge_df, label_name, split_control, train_ratio, verbose=False)
    train_avg, test_avg = np.average(acc_arr, axis=0)

    return train_avg, test_avg


def get_classification_random_accuracy(merge_df, label_name='kmean_num_label', split_control=False, train_ratio=0.8,
                                       verbose=True):
    """
    merge_df = dataframe with a column `label_name` that contains the label we want to build the random statistics on
    """
    merge_df_tt = get_merged_tt_df(merge_df, split_control=False, train_ratio=0.8)
    df = merge_df_tt[merge_df_tt.tt == 'train']
    random_predict_func = get_classification_random_function(df, label_name=label_name)
    random_predict = random_predict_func(len(df))
    train_acc = accuracy_score(random_predict, df[label_name])

    df = merge_df_tt[merge_df_tt.tt == 'test']
    random_predict = random_predict_func(len(df))
    test_acc = accuracy_score(random_predict, df[label_name])
    if verbose:
        print("Random accuracy")
        print(f"Random train Accuracy {train_acc}")
        print(f"Random test Accuracy {test_acc}")
    return train_acc, test_acc


def plot_classification_scatter(clf, test_df, meta_idx, labels_mapping):
    X_test = test_df.iloc[:, :meta_idx].values
    test_pred = clf.predict(X_test)
    pred_proba = clf.predict_proba(X_test)
    max_proba = np.max(pred_proba, axis=1)
    test_df['max_proba'] = max_proba
    test_df['pred'] = test_pred
    # px.strip(test_df,x='kmeans_label',y='max_proba')

    test_df.pred = test_df.pred.map(lambda x: labels_mapping[x])
    test_df.kmeans_label = test_df.kmeans_label.map(lambda x: labels_mapping[x])
    fig = px.strip(test_df, x='pred', y='max_proba', color='kmeans_label',
                   custom_data=['sampleID', 'symptoms', 'visit_age_mo'],
                   category_orders={'pred': list(labels_mapping.values())}
                   )

    fig.update_traces(
        hovertemplate="<br>".join([
            "predict %{x}, probability: %{y}",
            "sampleID: %{customdata[0]}",
            "symptom: %{customdata[1]}",
            "real age: %{customdata[2]}"
        ])
    )
    return fig

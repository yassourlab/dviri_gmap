# %%
from typing import Callable, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from loguru import logger
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from dataclasses import dataclass, InitVar, field
# from pydantic.dataclasses import dataclass
from pydantic import BaseModel
from gmap_runner import GmapRunner

# from consts import (best_clf_params, DEFAULT_ST_SORTED_ARRAY, META_PATH,
#                              norm_l7_path, l7_path
#  )


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
# %%

# norm_l7_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-norm-l7.csv"
# l7_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-l7.csv"
# norm_l6_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-norm-l6.csv"
# l6_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-l6.csv"
# META_PATH = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/edited_metadata.csv"

norm_l6_project = "/mnt/g/My Drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/FeatureTableNew.tsv"
META_PATH = "/mnt/g/My Drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/metadataNew.tsv"

@dataclass
class TopFeatures:
    importance: InitVar[np.ndarray]
    indices: InitVar[np.ndarray]
    sorted_features: np.ndarray
    sorted_importance:Optional[np.ndarray] = field(init=False)

    def __post_init__(self, importance, indices):
        self.sorted_features = importance[indices]

    class Config:
        arbitrary_types_allowed = True


class MyTreeModel:
    def __init__(self, runner: GmapRunner):
        self.runner: GmapRunner = runner
        self.merged_predict_df: Optional[pd.DataFrame] = None
        self.model: Union[RandomForestClassifier, RandomForestRegressor, None] = None
        self._top_features: Optional[TopFeatures] = None

    @property
    def top_features(self):
        return self._top_features

    def get_top_features(self, k=20):
        assert self.model is not None, "Must first fit model"
        data_df = self.runner.get_data(remove_meta=True)
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1]
        features = data_df.columns[indices]
        top_features = features[:k]
        self._top_features = TopFeatures(importance, indices, features)
        return top_features.tolist()


class ClassificationRunner(MyTreeModel):
    best_params = {'random_state': 666,
                   'n_estimators': 1200,
                   'min_samples_split': 20,
                   'min_samples_leaf': 2,
                   #  'max_features': 'sqrt',
                   'max_depth': None,
                   'bootstrap': False}

    def __init__(self, runner: GmapRunner) -> None:

        super().__init__(runner)
        self.noise_func = None
        self.acc = dict()
        self.loss = dict()
        self.wanted_label = 'sample_time'
        # self.setup_training_data()

    # def foo(self,x):
    #     if len(x) > 1:
    #         print('wow')
    # def get_
    def predict(self, wanted_label: str = None, dropna=True):
        """ Train and predict on a regression model with the regress_params.
        Add the columns predict,label,loss to the dataframe and return it.
        """
        runner = self.runner
        wanted_label = self.wanted_label if wanted_label is None else wanted_label
        X, y = runner.get_tain_data(wanted_label=wanted_label)

        meta_idx = runner.get_meta_idx()
        merge_df = runner.get_data()

        if dropna:
            nan_map = y.isna()
            X = X[~nan_map]
            y = y[~nan_map]
            merge_df = merge_df.dropna(subset=[wanted_label], inplace=False)

        model = RandomForestClassifier(**self.best_params)
        logger.info(f"Fitting model to {X.shape} examples")
        model.fit(X, y)

        merged_training_df = merge_df
        predict_data = merged_training_df.iloc[:, :meta_idx].values
        predict = model.predict(predict_data)
        label = merged_training_df[wanted_label]

        merged_training_df = merged_training_df.assign(predict=predict, label=label)
        self.merged_predict_df = merged_training_df
        self.model = model
        return self.merged_predict_df

    def calc_loss(self, verbose=True):
        merged_predict_df = self.merged_predict_df
        for tt in ['train', 'test']:
            df = merged_predict_df[merged_predict_df.tt == tt]
            label = df.label
            predict = df.predict
            accuracy = accuracy_score(label, predict)
            self.acc[tt] = accuracy
            if verbose:
                logger.info(f"{tt} acc, loss: {self.acc[tt]}\t ")

    def get_classifier_accuracy(self, merge_df_tt, return_all=False):
        """get Classification classifier accuracy by training on the given dataframe.
        Args:
            split_tt: If True, will give the merge_df_tt a new tt assignment. Otherwise, assume that the DataFrame already
            has tt and uses it.
            return_all: If true, will return a dictionary with "test_pred", "train_pred", "merge_df_tt"
        Returns:
            train_acc,test_acc
        """
        if merge_df_tt is None:
            merge_df_tt = self.merged_predict_df
        else:
            assert 'tt' in merge_df_tt, "Must pass a df with 'tt' division if split_tt is False "

        train_df = merge_df_tt[merge_df_tt.tt == 'train']
        test_df = merge_df_tt[merge_df_tt.tt == 'test']

        train_acc = self.acc['train']
        test_acc = self.acc['test']
        test_df['pred'] = test_df.pred
        train_df['pred'] = train_df.pred
        if return_all:
            res_dict = {
                "test_pred": test_df.pred.values,
                "train_pred": train_df.pred.values,
                "merge_df_tt": merge_df_tt,
                "train_df": train_df,
                "test_df": test_df
            }
            return res_dict

        return train_acc, test_acc

    def plot_confusion_matrix(self, *args, **kwargs):
        ct, cnt = self.get_confusion_matrix(*args, **kwargs)
        c = 'brwnyl'
        ct = ct.round(3)
        fig = ff.create_annotated_heatmap(ct.to_numpy().T, x=ct.columns.tolist(), y=ct.columns.tolist(), colorscale=c)
        fig.update_layout(title_text='Confusion Table - Label(rows)/Pred(cols)', font=dict(size=18))
        fig['layout']['xaxis']['side'] = 'bottom'
        fig.show()

    def get_confusion_matrix(self, classes_arr=None, label_col='kmeans_label',
                             normalize='index', rows_as='label',
                             res_data=None):
        """
        test_df: DataFrame in the shape of (N,c) where N is the number of samples the prediciton was made on. Needs to have columns label_col
        pred: An array of a classifier prediction over which the confusion matrix will be. Need to be in shape (N,)
        classes_arr: An Array with the classes in the classification. The order the classes will be, is the order in the output table
        normalize : bool, {'all', 'index', 'columns'}, or {0,1}, default False
        Normalize by dividing all values by the sum of values.
        """
        #     if classes_arr is None:
        #         classes_arr = ['onemonth', 'twomonth','fourmonth', 'sixmonth', 'ninemonth' , 'oneyear']
        # label_col = 'symptoms'
        test_df = self.merged_predict_df.query('tt == "test"')
        test_df['pred'] = test_df.predict
        label_col = label_col if label_col is not None else self.wanted_label
        if res_data is None:
            res = self.get_classifier_accuracy(test_df, return_all=True)
        else:
            res = res_data
        test_pred = res['test_pred']
        test_df = res['test_df']

        # ct,cnt = get_confusion_matrix(test_df,test_df.pred,test_df[label_col].unique().tolist(), label_col=label_col, return_count=True, normalize = 'index')
        pred = test_df.pred
        classes_arr = test_df[label_col].unique().tolist() if classes_arr is None else classes_arr

        df_mapping = pd.DataFrame({label_col: classes_arr})
        sort_mapping = df_mapping.reset_index().set_index(label_col)

        pred_merge_df = test_df.copy()
        pred_merge_df['pred_names'] = pred
        pred_merge_df['label_num'] = pred_merge_df[label_col].map(sort_mapping['index'])
        #     pred_merge_df['kmeans_label'].unique(),pred_merge_df['sample_time_num'].unique()
        pred_merge_df['pred_names_num'] = pred_merge_df['pred_names'].map(sort_mapping['index'])

        label_s = pd.Series(pred_merge_df.label_num, name='label')
        pred_s = pd.Series(pred_merge_df.pred_names_num, name='pred')

        if rows_as == 'predict':
            ct = pd.crosstab(pred_s, label_s, normalize=normalize)
            cnt = pd.crosstab(pred_s, label_s)

            mapping_dict = df_mapping.to_dict()[label_col]
            ct = ct.rename(columns=mapping_dict, index=mapping_dict)
            cnt = cnt.rename(columns=mapping_dict, index=mapping_dict)

            # If prediction is missing values, add them here with 0 for count and success
            missing_rows = list(set(classes_arr) - set(cnt.index))
            for missing_row in missing_rows:
                empty_row = pd.Series({k: 0 for k in ct.columns.tolist()}, name=missing_row)
                ct = ct.append(empty_row, ignore_index=False)
                cnt = cnt.append(empty_row, ignore_index=False)


        elif rows_as == 'label':
            ct = pd.crosstab(label_s, pred_s, normalize=normalize)
            cnt = pd.crosstab(label_s, pred_s)

            mapping_dict = df_mapping.to_dict()[label_col]
            ct = ct.rename(columns=mapping_dict, index=mapping_dict)
            cnt = cnt.rename(columns=mapping_dict, index=mapping_dict)

            missing_cols = list(set(classes_arr) - set(cnt.columns))
            # If prediction is missing values, add them here with 0 for count and success

            for missing_col in missing_cols:
                ct = ct.assign(**{missing_col: 0})
                cnt = cnt.assign(**{missing_col: 0})
        #             empty_row = pd.Series({k:0 for k in ct.columns.tolist()},name=missing_col)
        #             ct = ct.append(empty_row, ignore_index=False)
        #             cnt = cnt.append(empty_row, ignore_index=False)

        else:
            raise ValueError(
                f"Rows as parameter got an unknown argument. possibilities are predict \ label but got {rows_as}")
        ct = ct.loc[sorted(ct, key=classes_arr.index), sorted(ct, key=classes_arr.index)]
        cnt = cnt.loc[sorted(cnt, key=classes_arr.index), sorted(cnt, key=classes_arr.index)]
        return ct, cnt

    def plot_classification_scatter(self, labels_mapping):
        meta_idx = self.runner.get_meta_idx()

        test_df = self.merged_predict_df.query('tt == "test"')
        X_test = test_df.iloc[:, :meta_idx].values
        test_pred = test_df.predict.values
        pred_proba = self.model.predict_proba(X_test)
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

    def get_classification_random_function(self, merged_training_df, label_name='sample_time', n_bins=50):
        meta_idx = self.runner.get_meta_idx
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


class RegressionRunner(MyTreeModel):
    best_params = {
        'random_state': 666,
        'n_estimators': 1800,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'max_depth': 100,
        'bootstrap': False}

    def __init__(self, runner: GmapRunner) -> None:
        self.runner = runner
        runner.get_data()
        self.merged_predict_df = None
        self.noise_func = None
        self.acc = dict()
        self.loss = dict()

        # self.setup_training_data()

    # def foo(self,x):
    #     if len(x) > 1:
    #         print('wow')
    # def get_
    def calc_loss(self, err_margin):
        merged_predict_df = self.merged_predict_df
        for tt in ['train', 'test']:
            df = merged_predict_df[merged_predict_df.tt == tt]
            label = df.label
            predict = df.predict
            loss_arr = abs(predict - label)
            self.acc[tt] = np.count_nonzero(loss_arr < err_margin) / len(loss_arr)
            self.loss[tt] = np.mean(loss_arr)
            logger.info(f"{tt} acc, loss: {self.acc[tt]}\t {self.loss[tt]}")

    def predict(self):
        """ Train and predict on a regression model with the regress_params. 
        Add the columns predict,label,loss to the dataframe and return it.
        """
        runner = self.runner
        X, y = runner.get_tain_data()

        meta_idx = runner.get_meta_idx()
        merge_df = runner.get_data()

        regression_model = RandomForestRegressor(**self.best_params)
        # if add_noise:
        #     d = np.random.dirichlet(np.full(X.shape[1],1/10),X.shape[0])
        #     X = (X+d)/2 # each row in d,X sums to 1. Re-normalize to make it sum to 1 after addition
        logger.info(f"Fitting model to {X.shape} examples")
        regression_model.fit(X, y)

        merged_training_df = merge_df
        cols = merged_training_df.columns.tolist()

        predict = regression_model.predict(merged_training_df.iloc[:, :meta_idx].values)
        label = merged_training_df.visit_age_mo
        loss_arr = abs(predict - label)
        merged_training_df = merged_training_df.assign(predict=predict, label=label, loss=loss_arr)
        self.merged_predict_df = merged_training_df
        return self.merged_predict_df

    def get_random_function(self, merged_training_df, n_bins=50):
        meta_idx = self.runner.get_meta_idx
        X = merged_training_df.iloc[:, :meta_idx].values
        label = merged_training_df.visit_age_mo

        values, bins = np.histogram(label, bins=n_bins)
        prob = values / np.sum(values)
        return lambda size: np.random.choice(bins[1:], size=size, p=prob)

    def draw_regresssion_line(self, fig, regr_data, name='regression', pred_name='predict', color=None):
        """Draw a regression line on a figure

        Args:
            fig ([type]): An active figure to add the regression line to
            regr_data ([type]): The df to perform the regression on. 
            name (str, optional): [description]. Defaults to 'regression'.
            pred_name (str, optional): [description]. Defaults to 'predict'.
            color ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: The new figure with the regression line
        """
        regr, X = self.get_linear_regressor(regr_data, pred_name, return_data=True)
        reg_predict = regr.predict(X)
        fig.add_trace((go.Scatter(x=X[:, 0], y=reg_predict,
                                  mode='lines',
                                  name=name,
                                  marker={"color": color},
                                  line=dict(width=3))))
        return fig

    def get_linear_regressor(self, regr_data, pred_name, return_data=False):
        #     test_map = merged_training_df.tt == 'test'
        meta_idx = self.runner.get_meta_idx()

        # training_data = merged_training_df.iloc[:, :meta_idx].loc[test_map]
        assert len(regr_data) > 0, "test map didn't return any data point"
        predict = regr_data[pred_name]
        #     predict = best_model.predict(merged_training_df.iloc[:,:meta_idx].loc[test_map].values)
        label = regr_data.visit_age_mo.values
        # X = np.stack([label,predict]).T
        X = label.reshape(-1, 1)
        # sort_idxs= np.argsort(label)
        y = predict
        # Create linear regression object

        regr = linear_model.LinearRegression()
        regr.fit(X, y)
        if return_data:
            return regr, X

        return regr

    def draw_regression_results(self, draw_lines=True, show=True):
        """Draw the result of the regression. Must first call the predict method

        Args:
            draw_lines (bool, optional): [description]. Defaults to True.
            show (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        merged_training_df = self.merged_predict_df
        merged_training_df.loc[pd.isna(merged_training_df.symptoms), 'symptoms'] = 'unknown'
        fig = px.scatter(merged_training_df, x='visit_age_mo', y='predict', color='tt',
                         custom_data=['symptoms', 'sampleID'])

        if draw_lines:
            for data in fig['data']:
                legendgroup = data['legendgroup']
                regr_data = merged_training_df[merged_training_df.tt == legendgroup]
                self.draw_regresssion_line(fig, regr_data, f'{legendgroup} regression', color=data['marker']['color'])

        if show:
            fig.show()
        return fig


class Randomizer:
    def __init__(self, name: str, randomize_dict: dict, chance_deactivate=0.3):
        self.randomize_dict = randomize_dict
        self.chance_deactivate = chance_deactivate
        self.name = name

    def activate(self, method: Callable, verbose=True):
        to_activate = np.random.rand() > self.chance_deactivate
        if not to_activate:
            if verbose:
                logger.info(f"No random for {self.name}")
            return

        params_dict = dict()
        for param, param_options in self.randomize_dict.items():
            params_dict[param] = self._set_random_param(param_options)

        if verbose:
            logger.info(f"{self.name} Random values: {params_dict}")
        method(**params_dict)

    def _set_random_param(self, param_options: list):
        option_idx = np.random.randint(0, len(param_options))
        return param_options[option_idx]


rand_interp_dict = dict(
    num_extra_samples=np.arange(1, 8),
    interp_interval=np.arange(0.05, 0.9, 0.05),
    noise_type=['interpulate']

)
rand_label_noise_dict = dict(
    loc=[0],
    scale=np.arange(0.1, 1.0, 0.1),
    noise_type=['label']
)


def draw_results(logger_path):
    with open(logger_path, 'r') as f:
        file_txt = f.read()

    import re
    train_acc, train_loss = np.array(re.findall(r'train acc.* ([0-9]*[.]\d*)\s*([0-9]*[.]\d*)', file_txt)).T.astype(
        float).round(3)
    test_acc, test_loss = np.array(re.findall(r'test acc.* ([0-9]*[.]\d*)\s*([0-9]*[.]\d*)', file_txt)).T.astype(
        float).round(3)
    x = np.arange(0, len(test_acc))
    # fig = plt.plot(x, test_acc,"ro",x,train_acc,"b")

    # plt.plot(x, test_acc, "ro",0,test_acc[0],'bo')
    plt.plot(test_loss, test_acc, "ro", test_loss[0], test_acc[0], 'bo')
    plt.plot(train_acc, test_acc, "ro", train_acc[0], test_acc[0], 'bo')

    # plt.yscale('log')
    plt.show()

    """
        2021-07-26 11:07:28.853 | INFO     | __main__:activate:657 - interp Random values: {'num_extra_samples': 1, 'interp_interval': 0.2, 'noise_type': 'interpulate'}
    2021-07-26 11:07:30.471 | INFO     | __main__:activate:657 - label Random values: {'loc': 0, 'scale': 0.9, 'noise_type': 'label'}
    2021-07-26 11:07:30.472 | INFO     | __main__:activate:649 - No random for label2
    2021-07-26 11:07:30.472 | INFO     | __main__:predict:549 - Fitting model to (2616, 78) examples
    2021-07-26 11:07:42.233 | INFO     | __main__:calc_loss:531 - train acc, loss: 0.9895287958115183	 0.3161243251148097
    2021-07-26 11:07:42.234 | INFO     | __main__:calc_loss:531 - test acc, loss: 0.45789473684210524	 1.5470459900605174
    """


def random_search_augmentations():
    """ Performs a random search on multiple autmentaitons options to get the best parameters to maximize the accuracy

    """
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_path = f"./logs/logging_{ts}.log"
    logger.add(logger_path)
    runner = GmapRunner(norm_l6_project, split_control=False, train_ratio=0.8)
    runner.init_data('visit_age_mo')
    inter_rand = Randomizer("interp", rand_interp_dict, chance_deactivate=0.1)
    label_noise_rand = Randomizer("label", rand_label_noise_dict, chance_deactivate=0.3)
    label_noise_rand2 = Randomizer("label2", rand_label_noise_dict, chance_deactivate=0.85)
    for i in range(10):
        regr_runner = RegressionRunner(runner)

        if i == 0:
            logger.info("No augmentations")
            regr_runner.predict()
            regr_runner.calc_loss(1)
            continue

        inter_rand.activate(runner.add_training_noise)
        label_noise_rand.activate(runner.add_training_noise)
        label_noise_rand2.activate(runner.add_training_noise)

        regr_runner.predict()
        regr_runner.calc_loss(1)

    draw_results(logger_path)


def main_old():
    runner = GmapRunner(norm_l7_path, split_control=False, train_ratio=0.8)
    runner.init_data()

    runner.update_merge_df(rows_map=runner.merge_df.sample_time != 'sick', split_tt=True)

    classes_arr = ['initial', 'year']
    # classes_arr = DEFAULT_ST_SORTED_ARRAY
    col_name = 'kmeans_label'

    # Get new labels
    merge_df_tt = runner.create_class_by_kmeans(classes_arr, col_name)
    meta_idx = runner.meta_idx

    clf = RandomForestClassifier(**best_clf_params)
    # pred_merge_df = merge_df_tt.loc[merge_df_tt.tt == 'test'].copy()
    train_df = merge_df_tt[merge_df_tt.tt == 'train']
    X = train_df.iloc[:, :meta_idx].values
    y = train_df.kmeans_label.values
    clf.fit(X, y)

    test_df = merge_df_tt[merge_df_tt.tt == 'test']
    X_test = test_df.iloc[:, :meta_idx].values
    y_test = test_df.kmeans_label.values
    pred = clf.predict(X_test)
    N_classes = len(clf.classes_)

    print(f"Got test accuracy {accuracy_score(y_test, pred)}")

    ct, cnt = get_confusion_matrix(test_df, pred, cls_wanted='sample_time', classes_arr=classes_arr, return_count=True)
    c = 'brwnyl'
    fig = ff.create_annotated_heatmap(ct.to_numpy().T, x=ct.columns.tolist(), y=ct.columns.tolist(), colorscale=c)
    fig.update_layout(title_text='Confusion Table - Pred(rows)/Label(cols)')
    fig['layout']['xaxis']['side'] = 'bottom'
    fig.show()

    fig = ff.create_annotated_heatmap(cnt.to_numpy().T, x=cnt.columns.tolist(), y=cnt.columns.tolist(), colorscale=c)
    fig.update_layout(title_text='Confusion Table - Pred(rows)/Label(cols)')
    fig['layout']['xaxis']['side'] = 'bottom'
    fig.show()

    # Split to 2 categories
    two_cls_pred = list(map(cat_to_bin(classes_arr), pred))
    two_cls_y_test = list(map(cat_to_bin(classes_arr), y_test))
    print(f"Got test accuracy {accuracy_score(two_cls_y_test, two_cls_pred)}")

    # test_df = pd.DataFrame().assign(pred=pred,bin_pred=two_cls_pred,bin_label=two_cls_y_test) #type: pd.DataFrame
    pred_df = pd.DataFrame().assign(pred=pred, bin_pred=two_cls_pred, bin_label=two_cls_y_test)  # type: pd.DataFrame
    t = pred_df.melt(id_vars=['bin_label'], value_vars=['pred'], value_name='predict')
    tbl = t.groupby(['bin_label', 'predict'])['variable'].agg('count').unstack().fillna(0)
    col_map = {n: f'{n} ({cat_to_bin(classes_arr)(n)})' for n in tbl.columns}
    tbl = tbl.reindex(DEFAULT_ST_SORTED_ARRAY, axis=1)
    tbl = tbl.rename(columns=col_map)

    print(tbl.div(tbl.sum(axis=1), axis=0))
    print(tbl.astype(int))

    # sorted(tbl,key=lambda x:classes_arr.index(x))

    # test_df.pivot_table()
    # dummy = pd.get_dummies(t['value'],prefix='pred')
    # t = pd.concat([t,dummy],axis=1)
    # gb = t.groupby(['two_cls_pred', 'value']).agg('count')
    # t.melt(id_vars=['two_cls_pred'],value_vars=[dummy.columns])


def cat_to_bin(classes_arr):
    return lambda x: 'onemonth' if classes_arr.index(x) < 3 else 'oneyear'


def main():
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger_path = f"./logs/logging_{ts}.log"
    logger.add(logger_path)
    runner = GmapRunner(norm_l6_project, split_control=False, train_ratio=0.8)
    runner.init_data('symptoms')
    inter_rand = Randomizer("interp", rand_interp_dict, chance_deactivate=0.1)
    label_noise_rand = Randomizer("label", rand_label_noise_dict, chance_deactivate=0.3)
    label_noise_rand2 = Randomizer("label2", rand_label_noise_dict, chance_deactivate=0.85)

    clf_runner = ClassificationRunner(runner)
    clf_runner.predict()
    clf_runner.calc_loss()

    meta_df = runner.get_data()
    meta_idx = runner.get_meta_idx()
    best_featuers = [28, 24, 65, 66, 34, 23, 22]  # From paper
    filt_meta_idx = len(best_featuers)
    filt_df = meta_df.iloc[:, best_featuers]
    filt_df = pd.concat([filt_df, meta_df.iloc[:, meta_idx:]], axis=1)
    allergy = 'allergy'
    filt_df.symptoms = filt_df.symptoms.map(
        {'Symptomatic': allergy, 'Pre-symptoms': allergy, 'Control': 'Control', 'Resolved': 'Control'})

    runner.set_data_df(filt_df, filt_meta_idx, True)
    # runner.add_training_noise(noise_type='label', loc=0, scale=0.9)
    _ = clf_runner.predict()
    clf_runner.calc_loss()

    runner.add_training_noise('interpulate', num_extra_samples=3, interp_interval=0.1)
    classes_arr = ['initial', 'middle', 'year']
    _ = runner.create_class_by_kmeans(classes_arr, new_col_name='kmeans_label')
    _ = clf_runner.predict(wanted_label='kmeans_label')
    clf_runner.calc_loss()

    ct, cnt = clf_runner.get_confusion_matrix()
    num_labels = ['0-4', '5-9', '9-14']
    num_labels = ['0-1', '2', '4', '6', '9', '12']
    classes_arr = ['onemonth', 'twomonth', 'fourmonth', 'sixmonth', 'ninemonth', 'oneyear']
    labels_mapping = {k: f"{k}({num_labels[i]})" for i, k in enumerate(classes_arr)}
    fig = clf_runner.plot_classification_scatter(labels_mapping=labels_mapping)
    fig.show()
    runner.reset_data_df()
    clf_runner.predict()
    clf_runner.calc_loss()
    # for i in range(1000):
    #     regr_runner = RegressionRunner(runner)
    #
    #     if i == 0:
    #         logger.info("No augmentations")
    #         regr_runner.predict()
    #         regr_runner.calc_loss(1)
    #         continue
    #
    #     inter_rand.activate(runner.add_training_noise)
    #     label_noise_rand.activate(runner.add_training_noise)
    #     label_noise_rand2.activate(runner.add_training_noise)
    #
    #     regr_runner.predict()
    #     regr_runner.calc_loss(1)
    #
    # draw_results(logger_path)


if __name__ == '__main__':
    # random_search_augmentations()
    main()

    # from plotnine import ggplot, aes, geom_boxplot, geom_boxplot, geom_point, position_jitter, scale_color_brewer, \
    #     scale_fill_brewer, theme_bw, theme, xlab
    # from plotnine import data as nintdata
    # from statannot import add_stat_annotation
    # interst_df = pd.read_csv("./py/interst.csv")
    # # tdf = interst_df[interst_df.groups == 'sick']
    # colors = ['Blues', 'BuGn', 'BuPu', 'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd', 'PuBu', 'PuBuGn', 'PuRd',
    #           'Purples', 'RdPu', 'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']
    # # ggplot(diamonds, aes(x='cut',y='price')) + geom_boxplot()
    # res_gg = ggplot(interst_df, aes(x='group', y='CPM', color='group', fill='group')) \
    #          + geom_boxplot() \
    #          + geom_point(position=position_jitter(width=0.2), size=2, alpha=0.8) \
    #          + scale_color_brewer(platte='Set1') \
    #          + theme_bw() \
    #          + xlab('Infection/disease status') \
    #     # + theme(legend_position = "NA")
    # # + scale_fill_brewer(palette = colors[15])
    # # p10
    # fig, p = res_gg.draw(return_ggplot=True, show=False)
    # axs = fig.get_axes()
    # axs[0].draw()
    # plt.show()
# %%

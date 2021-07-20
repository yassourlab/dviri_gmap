# %%
import matplotlib.pyplot as plt
from loguru import logger
from scipy import interpolate
import numpy as np
import pandas as pd
from functools import partial
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from plotly.subplots import make_subplots
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# from consts import (best_clf_params, DEFAULT_ST_SORTED_ARRAY, META_PATH,
#                              norm_l7_path, l7_path
                            #  )
from scipy.spatial import distance
import skbio
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
# %%

# norm_l7_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-norm-l7.csv"
# l7_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-l7.csv"
# norm_l6_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-norm-l6.csv"
# l6_path = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/feature-table-l6.csv"
# META_PATH = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/edited_metadata.csv"

norm_l6_project = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/FeatureTableNew.tsv"
META_PATH = "/mnt/c/Users/dviri/google_drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/metadataNew.tsv"

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
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = np.linspace(10, 100, num=5, dtype=int).tolist() + [None]
    # Minimum number of samples required to split a node
    min_samples_split = [5, 10, 20, 40]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [2, 4, 6, 10]
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


def draw_regresssion(merged_training_df, test_map, fig, name='regression', pred_name='predict'):
    regr, X = get_linear_regressor(merged_training_df, test_map, pred_name, return_data=True)
    reg_predict = regr.predict(X)
    fig.add_trace((go.Scatter(x=X[:, 0], y=reg_predict,
                              mode='lines',
                              name=name,
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
    df_enum = ["control_train_df", "control_test_df", "test_df"]
    for i in range(len(loss_lst)):
        print(df_enum[i])
        print(f"AVG loss = {(np.sum(loss_lst[i]) / len(acc_lst[i])).round(2)}")
        print(f"AVG acc = {(np.sum(acc_lst[i]) / len(acc_lst[i])).round(2)}")
        print(f"MIN acc = {np.min(acc_lst[i]).round(2)}")
        print(f"MAX acc = {np.max(acc_lst[i]).round(2)}")
        print()


def get_gmap_data(data_path: str):
    """
    return merge_df,meta_idx
    """
    otu_data = pd.read_csv(data_path, sep='\t', index_col=['OTU ID'])
    examples_data = otu_data.T
    metadata = pd.read_csv("./data/edited_metadata.csv", sep='\t')
    merge_df = examples_data.merge(metadata, left_index=True, right_on=['sampleID'])
    merge_df = merge_df.assign(is_control=(merge_df.symptoms == 'Control'))
    meta_idx = examples_data.shape[1]
    return merge_df, meta_idx


def get_confusion_matrix(df, pred, cls_wanted, classes_arr=None, return_count=False):
    """

    Args:
        df:
        pred:
        classes_arr: Array of classes in the way you want it to be sorted. if not passed, will use python default sort
        fir the classes

    Returns:

    """
    if classes_arr is None:
        # classes_arr = DEFAULT_ST_SORTED_ARRAY
        classes_arr = df[cls_wanted].unique().astype(list)

    df_mapping = pd.DataFrame({cls_wanted: classes_arr})
    sort_mapping = df_mapping.reset_index().set_index(cls_wanted)

    pred_merge_df = df.copy()
    pred_merge_df['pred_names'] = pred
    num_cls_name = f'{cls_wanted}_num'

    pred_merge_df[num_cls_name] = pred_merge_df['kmeans_label'].map(sort_mapping['index'])
    pred_merge_df['kmeans_label'].unique(), pred_merge_df[num_cls_name].unique()
    pred_merge_df['pred_names_num'] = pred_merge_df['pred_names'].map(sort_mapping['index'])

    pred_s = pd.Series(pred_merge_df.sample_time_num, name='pred')
    label_s = pd.Series(pred_merge_df.pred_names_num, name='label')

    ct = pd.crosstab(label_s, pred_s, normalize='index')
    mapping_dict = df_mapping.to_dict()[cls_wanted]
    ct = ct.rename(columns=mapping_dict, index=mapping_dict)

    if not return_count:
        return ct

    cnt = pd.crosstab(label_s, pred_s)
    mapping_dict = df_mapping.to_dict()[cls_wanted]
    cnt = cnt.rename(columns=mapping_dict, index=mapping_dict)
    return ct, cnt


class GmapRunner:
    def __init__(self, data_path, split_control=False, train_ratio=0.85, metadata_path=META_PATH):
        self.data_path = data_path
        self.split_control = split_control
        self.merge_df = None  # type: pd.DataFrame
        self.metadata_path = metadata_path
        self.train_ratio = train_ratio

    def init_data(self, split_control: bool = None, split_tt=True):
        split_control = self.split_control if split_control is None else split_control

        examples_data = pd.read_csv(self.data_path, sep='\t', index_col=['OTU ID']).T
        metadata = pd.read_csv(self.metadata_path, sep='\t')
        merge_df = examples_data.merge(metadata, left_index=True, right_on=['sampleID'])
        merge_df = merge_df.assign(is_control=(merge_df.symptoms == 'Control'))
        meta_idx = examples_data.shape[1]

        self.merge_df = merge_df
        self.meta_idx = meta_idx

        if split_tt:
            self.update_tt(split_control)

    def get_data(self):
        return self.merge_df

    def get_meta_idx(self):
        return self.meta_idx

    def update_tt(self, split_control=None, train_ratio=None):
        """
        Add tt division for the given dataframe with ratio. group by record_id to make sure same baby won't be both in train and test.
        """
        split_control = split_control if split_control is not None else self.split_control
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        self.merge_df = self.merge_df.groupby(['is_control', 'record_id']).apply(
            lambda x: split_train_test(x, split_control, train_ratio)).reset_index(drop=True)

    def update_merge_df(self, rows_map, split_tt=False):
        """

        Args:
            rows_map:
            split_tt: If set to True, will update the tt. It is recommended setting to True if removed rows

        Returns:

        """
        self.merge_df = self.merge_df.loc[rows_map]

        if split_tt:
            self.update_tt()

        return self.merge_df

    def create_class_by_kmeans(self, classes_arr, new_col_name='kmeans_label'):
        merge_df_tt = self.merge_df
        assert 'tt' in merge_df_tt, "Must first split to train/test before running create_class_by_kmeans"
        X = merge_df_tt.visit_age_mo.to_numpy().reshape(-1, 1)
        labels = merge_df_tt.sample_time.unique()
        kmeans = KMeans(n_clusters=len(classes_arr), random_state=666).fit(X)
        args_res = np.argsort(kmeans.cluster_centers_.reshape(-1))
        labels = list(map(lambda x: classes_arr[np.where(args_res == x)[0][0]], kmeans.labels_))
        # labels = list(map(lambda x: np.where(args_res == x)[0][0] ,kmeans.labels_))
        merge_df_tt = merge_df_tt.assign(**{new_col_name: labels})
        self.merge_df = merge_df_tt
        return merge_df_tt
    
    def pcoa_dim_reduction(self, i=1, j=2, k=3):
        df = self.merge_df
        # df = df.reset_index().rename(columns={'index': 'sample_name'})
        X = df.iloc[:, :self.meta_idx]
        Ar_dist = distance.squareform(distance.pdist(X, metric="braycurtis"))  # (m x m) distance measure
        DM_dist = skbio.stats.distance.DistanceMatrix(Ar_dist, ids=X.index)
        PCoA = skbio.stats.ordination.pcoa(DM_dist, number_of_dimensions=6)

        PCoA_samples_df = PCoA.samples
        dims = PCoA_samples_df.shape[1]

        PCoA_samples_df = PCoA.samples
        dims = PCoA_samples_df.shape[1]
        col = 'symptoms'
        # col = 'record_id'

        embedded_X = PCoA_samples_df.rename(columns={f"PC{i}": 'x', f"PC{j}": 'y', f"PC{k}": 'z'})
        plot_df = pd.concat([df, embedded_X], axis=1)

        embedded_X = embedded_X.assign(**{col: df[col]})
        # plot_df.record_id = plot_df.record_id.astype(str)
        # clean_df = plot_df[~plot_df[col].isna()].copy()
        embedded_X = embedded_X[~embedded_X[col].isna()].copy()
        fig = px.scatter_3d(embedded_X, x='x', y='y', z='z', color=col)
        return fig


class RegressionRunner:
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

        self.setup_training_data()
        

    def setup_training_data(self):
        runner = self.runner
        meta_idx = runner.get_meta_idx()
        merge_df = runner.get_data()
        self.setup_xy_train(merge_df,meta_idx)
        

    def setup_xy_train(self,merge_df,meta_idx):
        assert 'tt' in merge_df, "Must first split to tt before using this function"
        train_df = merge_df[merge_df.tt == 'train']
        self._train_X = train_df.iloc[:, :meta_idx].values
        self._train_y = train_df.visit_age_mo
        
    def add_training_noise(self, noise_type,add_data=True, **kwargs):
        X = self._train_X.copy()
        y = self._train_y.copy()
        if noise_type == 'normal':
            noise =  np.random.normal(**kwargs,size=X.shape)
            X = (X + noise)
            X = X/X.sum(axis=0,keepdims=1)
        elif noise_type == 'label':
            noise =  np.random.normal(**kwargs,size=y.shape)
            y += noise
            y = np.maximum(y,0)
        elif noise_type == 'interpulate':
            runner = self.runner
            meta_idx = runner.get_meta_idx()
            merge_df = runner.get_data()
            # merge_df.groupby(['record_id','visit_age_mo']).apply(self.foo)
            train_df = merge_df[merge_df.tt == 'train']
            gb_records_symptoms = train_df.groupby(['record_id','symptoms'])
            interpulate_df = gb_records_symptoms.apply(lambda x: self.interpulate_group(x,**kwargs))
            add_data = False
            self.setup_xy_train(interpulate_df,meta_idx)
            return

        if add_data:
            self._train_X = np.concatenate([self._train_X,X])
            self._train_y = np.concatenate([self._train_y,y])
        else:
            self._train_X = X
            self._train_y = y
    
    # def foo(self,x):
    #     if len(x) > 1:
    #         print('wow')

    def interpulate_group(self,group_df,num_extra_samples = 1, interp_interval=0.1,kind='cubic'):
        meta_idx = self.runner.meta_idx
        group_df = group_df.sort_values(['visit_age_mo'],ascending=True)
        data_df: pd.DataFrame = group_df.iloc[:,:meta_idx]
        data = data_df.to_numpy()

        if len(data) < 2:
            return group_df
        y = group_df.visit_age_mo.to_numpy()
        x = np.arange(0,data.shape[1],1)

        
        f = interpolate.interp2d(x,y,data,kind="linear")
        pd.DataFrame(data)

        ynew = np.arange(y.min(),y.max()+interp_interval,interp_interval).round(1)
        xnew = np.arange(0,data.shape[1],1)
        interp_data = f(xnew,ynew)
        interp_data = np.where(np.isnan(interp_data),0,interp_data)
        # print(data.shape, interp_data.shape, x.shape,y.shape,xnew.shape,ynew.shape)
        inter_df = pd.DataFrame(interp_data.round(3))
        inter_df.columns = group_df.iloc[:,:meta_idx].columns
        inter_df['visit_age_mo'] = ynew

        # cols = list(inter_df.columns)
        # overlapping_df = inter_df.loc[inter_df.visit_age_mo.isin(group_df.visit_age_mo)]
        # group_overlap_df =  group_df.loc[group_df.visit_age_mo.isin(overlapping_df.visit_age_mo)]
        # overlapping_df.iloc[:,:meta_idx] = group_overlap_df.iloc[:,:meta_idx]
        new_interpulation_rows_df = inter_df[~inter_df.visit_age_mo.isin(group_df.visit_age_mo)] #remove original values
        new_interpulation_rows_df['source'] = 'interp'
        group_df['source'] = 'data'
        res_df = pd.concat([new_interpulation_rows_df,group_df])
        res_df = res_df.sort_values(['visit_age_mo']).reset_index(drop=True)
        idxs = res_df[res_df.source == 'data'].index.tolist()
        take_idxs = list()
        for i in range(1,len(idxs)-1):
            idx_range = idxs[i] - idxs[i-1]
            interval = idx_range//(num_extra_samples+1)
            if interval == 0:
                continue
            range_idxs = list(range(idxs[i-1],idxs[i],interval))[1:]
            take_idxs += range_idxs
        
        res_df.tt = res_df.tt.dropna().unique()[0]

                # ranges = orig_data_idxs[1:]-orig_data_idxs[:-1]
        # intervals = ranges//(num_extra_samples+1)
        # mul_range = np.arange(1,num_extra_samples+1)[np.newaxis, :]
        # samples_jumps = np.tile(mul_range,(len(ranges),1))
        # take_idxs = samples_jumps * intervals[np.newaxis,:].T
        # take_idxs = np.concatenate([take_idxs.ravel(),orig_data_idxs])
        # final_df = res_df.iloc[take_idxs].sort_values(['visit_age_mo']).reset_index(drop=True)
        final_df = res_df.iloc[list(set(take_idxs+idxs))].sort_values(['visit_age_mo']).reset_index(drop=True)
        self.draw_interp_results(final_df)
        return final_df
    
    def draw_interp_results(self, df):
        
        data = df.iloc[:,:self.runner.get_meta_idx()]
        orig_data_full = df[df.source == 'data']
        orig_data = orig_data_full.iloc[:,:self.runner.get_meta_idx()]
        sort_arg = np.argsort(data.sum(axis=0))
        data:pd.DataFrame = data.iloc[:,sort_arg[::-1]].reset_index(drop=True)
        orig_data:pd.DataFrame = orig_data.iloc[:,sort_arg[::-1]].reset_index(drop=True)
        data_np = data.to_numpy()
        orig_data_np = orig_data.to_numpy()
        plot = plt.plot(df.visit_age_mo,data_np[:,0],'ro-')
        plt.show()
        plot = plt.plot(orig_data_full.visit_age_mo,orig_data_np[:,0],'bo-')
        plt.show()
        plot = plt.plot(df.visit_age_mo,data_np[:,0],'ro-',orig_data_full.visit_age_mo,orig_data_np[:,0],'b-')
        plt.show()
        print()

    # def get_
    def calc_loss(self,err_margin):
        merged_predict_df = self.merged_predict_df
        for tt in ['train','test']:
            df = merged_predict_df[merged_predict_df.tt == tt]
            label = df.label
            predict = df.predict
            loss_arr = abs(predict-label)
            self.acc[tt] = np.count_nonzero(loss_arr < err_margin) / len(loss_arr)
            self.loss[tt] =  np.mean(loss_arr)
            print(f"{tt} acc, loss:\n", self.acc[tt],self.loss[tt])

    def predict(self,err_margin=0.1):
        """ Train and predict on a regression model with the regress_params. 
        Add the columns predict,label,loss to the dataframe and return it.
        """
        runner = self.runner
        X = self._train_X
        y = self._train_y

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
        loss_arr = abs(predict-label)
        merged_training_df = merged_training_df.assign(predict=predict, label=label, loss=loss_arr)
        self.merged_predict_df = merged_training_df
        return self.merged_predict_df

    def get_random_function(self, merged_training_df, n_bins=50):
        meta_idx = self.runner.get_meta_idx
        X = merged_training_df.iloc[:, :meta_idx].values
        label = merged_training_df.visit_age_mo

        values, bins = np.histogram(label, bins=n_bins)
        prob = values/np.sum(values)
        return lambda size: np.random.choice(bins[1:], size=size, p=prob)

    def draw_regresssion_line(self,fig, regr_data, name='regression',pred_name = 'predict', color=None):
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
        regr,X = self.get_linear_regressor(regr_data,pred_name,return_data=True)
        reg_predict = regr.predict(X)
        fig.add_trace((go.Scatter(x=X[:,0], y=reg_predict,
                            mode='lines',
                            name=name,
                                marker={"color":color},
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
        merged_training_df.loc[pd.isna(merged_training_df.symptoms),'symptoms']='unknown'
        fig = px.scatter(merged_training_df,x='visit_age_mo',y='predict',color='tt',custom_data=['symptoms','sampleID'])

        if draw_lines:
            for data in fig['data']:
                legendgroup = data['legendgroup']
                regr_data = merged_training_df[merged_training_df.tt ==legendgroup]
                self.draw_regresssion_line(fig,regr_data,f'{legendgroup} regression',color=data['marker']['color'])

        if show:
            fig.show()
        return fig

def main():
    runner = GmapRunner(norm_l6_project, split_control=False, train_ratio=0.8)
    runner.init_data()

    for i in range(6):
        logger.info(f"Predicting for i {i}")
        regr_runner = RegressionRunner(runner)
        
        regr_runner.add_training_noise("interpulate",num_extra_samples=3, interp_interval=0.1*(i+1))
        regr_runner.predict()
        regr_runner.calc_loss(1)
    
    for i in range(5):
        logger.info(f"Predicting for i {i}")
        regr_runner = RegressionRunner(runner)
        # if i > 0:
            # regr_runner.add_training_noise("normal",loc=0,scale=0.05)
            # if i > 1:
            #     regr_runner.add_training_noise("label",loc=0,scale=0.5)
        
        regr_runner.add_training_noise("interpulate",num_extra_samples=i+1)
        regr_runner.predict()
        regr_runner.calc_loss(1)
            
        # regr_runner.draw_regression_results(draw_lines=True)
    pass
    # runner.update_merge_df(rows_map=runner.merge_df.sample_time != 'sick',split_tt=True)


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


if __name__ == '__main__':
    # main()
    main()

# %%

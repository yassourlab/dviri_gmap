from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import skbio
from matplotlib import pyplot as plt
from scipy import interpolate
from scipy.spatial import distance
from sklearn.cluster import KMeans

norm_l6_project = "/mnt/g/My Drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/FeatureTableNew.tsv"
META_PATH = "/mnt/g/My Drive/University/Masters/MoranLab/Dvir/GMAP/projects/py/nb/data/metadataNew.tsv"

from functions import split_train_test


class DataSep(Enum):
    All_Samples = 1
    OneSamplePerKidPerTime = 2
    samples_02_Model = 3
    last_Pre_symptoms = 4
    first_Symptomatic = 5
    first_Resolved = 6
    last_Resolved = 7
    samples_912_Model = 8


class GmapRunner:
    def __init__(self, data_path, split_control=False, train_ratio=0.85, metadata_path=META_PATH,
                 wanted_label: str = None):
        self.data_names = None
        self.data_path = data_path
        self.split_control = split_control
        self.merge_df: Optional[pd.DataFrame] = None
        self.meta_idx: Optional[int] = None
        self.__original_df: Optional[pd.DataFrame] = None
        self.__original_meta_idx: Optional[pd.DataFrame] = None

        self.metadata_path = metadata_path
        self.train_ratio = train_ratio
        self._train_X: Optional[np.ndarray] = None
        self._train_y: Optional[np.ndarray] = None
        self._data_sep: Optional[dict] = None
        self._origin_label = wanted_label
        self.label = wanted_label
        self._data_cols_idxs: Optional[list] = None

    @property
    def data_cols_idxs(self):
        if self._data_cols_idxs is None:
            self._data_cols_idxs = list(range(self.meta_idx))
        return self._data_cols_idxs

    def set_wanted_label(self, wanted_label):
        if self._origin_label is None:
            self._origin_label = wanted_label
        self.label = wanted_label

    def get_wanted_label(self):
        return self.label

    def init_data(self, wanted_label: str, split_control: bool = None, split_tt=True, tt_split_type=None):
        self.set_wanted_label(wanted_label)
        split_control = self.split_control if split_control is None else split_control

        examples_data = pd.read_csv(self.data_path, sep='\t', index_col=['OTU ID']).T
        metadata = pd.read_csv(self.metadata_path, sep='\t')
        merge_df = examples_data.merge(metadata, left_index=True, right_on=['sampleID'])
        merge_df = merge_df.assign(is_control=(merge_df.symptoms == 'Control'))
        meta_idx = examples_data.shape[1]
        self.merge_df = merge_df

        merge_df = self.update_tt(split_control, tt_split_type=tt_split_type)

        self.set_data_df(merge_df, meta_idx)

    def filter_data_paper(self, filter_type: DataSep):
        if self._data_sep is None:
            self._init_filter_data()

        if filter_type == DataSep.All_Samples:
            return

        assert filter_type in DataSep
        features = self._data_sep[filter_type.name]
        merge_df = self.get_data()
        merge_df = merge_df.loc[merge_df.sampleID.isin(features)]
        self.set_data_df(merge_df, self.get_meta_idx(), True)

    def _init_filter_data(self, f_path="./py/nb/data/data_seperat"):
        self._data_sep = dict()
        with open(f_path, 'r') as f:
            for line in f.readlines():
                name, *subjects_names = line.split(",")
                self._data_sep[name] = subjects_names

    def set_data_df(self, df, meta_idx, set_train_xy=True, wanted_label: str = None, x_meta_names: list = None,
                    data_idxs=None):
        if self.__original_df is None:
            self.__original_df = df
            self.__original_meta_idx = meta_idx

        self.merge_df = df
        self.meta_idx = meta_idx
        self._data_cols_idxs = data_idxs if data_idxs is not None else self.data_cols_idxs

        if set_train_xy:
            self.setup_xy_train(wanted_label=wanted_label, data_names=x_meta_names)

    def reset_data_df(self):
        """
        Restore original df with training data
        """
        self.set_wanted_label(self._origin_label)
        self.set_data_df(self.__original_df, self.__original_meta_idx)
        self._data_cols_idxs = None
        self.setup_xy_train()

    def get_data(self, remove_meta=False):
        if remove_meta:
            if self._data_cols_idxs is not None:
                return self.merge_df.iloc[:, self.data_cols_idxs]
            return self.merge_df.iloc[:, :self.meta_idx]

        return self.merge_df

    def get_meta_idx(self):
        return self.meta_idx

    def update_tt(self, split_control=None, train_ratio=None, tt_split_type='record_id'):
        """
        Add tt division for the given dataframe with ratio. group by record_id to make sure same baby won't be both in train and test.
        """

        split_control = split_control if split_control is not None else self.split_control
        train_ratio = train_ratio if train_ratio is not None else self.train_ratio
        if tt_split_type is None or tt_split_type == 'record_id':
            self.merge_df = self.merge_df.groupby(['is_control', 'record_id']).apply(
                lambda x: split_train_test(x, split_control, train_ratio)).reset_index(drop=True)
        elif tt_split_type == 'symptoms':
            self.merge_df['tt'] = np.nan
            for symptom in self.merge_df.symptoms.unique():
                num_elems = len(self.merge_df[self.merge_df.symptoms == symptom])
                rand_vals = np.random.rand(num_elems)
                perc = np.percentile(rand_vals, train_ratio * 100)
                tt_map = rand_vals>perc
                tt = ['test' if is_test else 'train' for is_test in tt_map]
                self.merge_df.loc[self.merge_df.symptoms == symptom, 'tt'] = tt
            self.merge_df.groupby(['symptoms']).apply(lambda x: np.random.rand(len(x)))
        return self.merge_df

    def update_merge_df(self, rows_map, split_tt=False, tt_split_type=None):
        """

        Args:
            rows_map:
            split_tt: If set to True, will update the tt. It is recommended setting to True if removed rows

        Returns:

        """
        self.merge_df = self.merge_df.loc[rows_map]

        if split_tt:
            self.update_tt(tt_split_type=tt_split_type)

        return self.merge_df

    def create_class_by_kmeans(self, classes_arr, new_col_name='kmeans_label'):
        merge_df_tt = self.merge_df
        assert 'tt' in merge_df_tt, "Must first split to train/test before running create_class_by_kmeans"
        X = merge_df_tt.visit_age_mo.to_numpy().reshape(-1, 1)
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

    def get_train_data(self, copy=False, wanted_label=None, x_meta_names: list = None):
        if self._train_X is None or wanted_label is not None or x_meta_names is not None:
            self.setup_xy_train(wanted_label, data_names=x_meta_names)
        if not copy:
            return self._train_X, self._train_y
        x_train = self._train_X.copy()
        y_train = self._train_y.copy()
        return x_train, y_train

    def set_train_data(self, *, X=None, y=None):
        if X is not None:
            self._train_X = X
        if y is not None:
            self._train_y = y

    def add_training_noise(self, noise_type, add_data=True, **kwargs):
        X, y = self.get_train_data(copy=True)
        if noise_type == 'normal':
            noise = np.random.normal(**kwargs, size=X.shape)
            X = (X + noise)
            X = X / X.sum(axis=0, keepdims=1)
        elif noise_type == 'label':
            noise = np.random.normal(**kwargs, size=y.shape)
            y += noise
            y = np.maximum(y, 0)
        elif noise_type == 'interpulate':
            runner = self
            meta_idx = runner.get_meta_idx()
            merge_df = runner.get_data()
            # merge_df.groupby(['record_id','visit_age_mo']).apply(self.foo)
            train_df = merge_df[merge_df.tt == 'train']
            test_df = merge_df[merge_df.tt == 'test']
            gb_records_symptoms = train_df.groupby(['record_id', 'symptoms'])
            interpulate_df = gb_records_symptoms.apply(lambda x: self.interpulate_group(x, **kwargs))
            merge_df = pd.concat([interpulate_df, test_df])
            add_data = False
            self.set_data_df(merge_df, meta_idx)
            # self.set_train_data(interpulate_df, meta_idx)
            return

        if add_data:
            self._train_X = np.concatenate([self._train_X, X])
            self._train_y = np.concatenate([self._train_y, y])
        else:
            self._train_X = X
            self._train_y = y

    def setup_xy_train(self, wanted_label=None, data_names: list = None):
        if wanted_label is not None:
            label = wanted_label
            self.set_wanted_label(label)
        else:
            label = self.get_wanted_label()
        meta_idx = self.get_meta_idx()
        merge_df = self.get_data()
        assert 'tt' in merge_df, "Must first split to tt before using this function"
        train_df = merge_df[merge_df.tt == 'train']
        train_X = train_df.iloc[:, :meta_idx]
        train_y = train_df[label]

        if data_names is not None:
            self.data_names = data_names
            # meta_data = train_df.loc[:, data_names]
            self._data_cols_idxs = list(range(meta_idx)) + [merge_df.columns.tolist().index(name) for name in
                                                            data_names]
            train_X = train_df.iloc[:, self.data_cols_idxs].values
        else:
            train_X = train_X.values
        self.set_train_data(X=train_X, y=train_y)

    def filter_by_abundance(self, threshold=1.0):
        """
        Filter columns if their train abundance sum is under the given threshold
        Args:
            threshold: Filter features with abundance lower than threshold

        Returns:

        """
        meta_idx = self.get_meta_idx()
        merge_df = self.get_data()
        assert 'tt' in merge_df, "Must first split to tt before using this function"
        train_df = merge_df[merge_df.tt == 'train']
        train_X = train_df.iloc[:, :meta_idx]
        over_threshold = np.where(train_X.sum(axis=0).astype(int) > threshold)[0]
        filt_cols_idxs = np.concatenate([over_threshold, np.arange(self.meta_idx, merge_df.shape[1])])
        filtered_df = merge_df.iloc[:, filt_cols_idxs]
        self.set_data_df(filtered_df, len(over_threshold), True, self.get_wanted_label(), self.data_names,
                         np.arange(len(over_threshold)))

    def interpulate_group(self, group_df, num_extra_samples=1, interp_interval=0.1, kind='cubic'):
        meta_idx = self.get_meta_idx()
        group_df = group_df.sort_values(['visit_age_mo'], ascending=True)
        data_df: pd.DataFrame = group_df.iloc[:, :meta_idx]
        data = data_df.to_numpy()

        if len(data) < 2:
            return group_df
        if len(data) < 4 and kind == 'cubic':
            kind = 'linear'
        y = group_df.visit_age_mo.to_numpy()
        x = np.arange(0, data.shape[1], 1)

        f = interpolate.interp2d(x, y, data, kind)
        pd.DataFrame(data)

        ynew = np.arange(y.min(), y.max() + interp_interval, interp_interval).round(1)
        xnew = np.arange(0, data.shape[1], 1)
        interp_data = f(xnew, ynew)
        interp_data = np.where(np.isnan(interp_data), 0, interp_data)
        # print(data.shape, interp_data.shape, x.shape,y.shape,xnew.shape,ynew.shape)
        inter_df = pd.DataFrame(interp_data.round(3))
        inter_df.columns = group_df.iloc[:, :meta_idx].columns
        inter_df['visit_age_mo'] = ynew
        inter_df['symptoms'] = group_df.symptoms.tolist()[0]

        new_interpulation_rows_df = inter_df[
            ~inter_df.visit_age_mo.isin(group_df.visit_age_mo)]  # remove original values
        new_interpulation_rows_df['source'] = 'interp'
        group_df['source'] = 'data'
        res_df = pd.concat([new_interpulation_rows_df, group_df])
        res_df = res_df.sort_values(['visit_age_mo']).reset_index(drop=True)
        final_df = self.extract_equale_interval_samples(num_extra_samples, res_df)

        # self.draw_interp_results(final_df)
        return final_df

    def extract_equale_interval_samples(self, num_extra_samples, res_df):
        idxs = res_df[res_df.source == 'data'].index.to_numpy()
        res_df.tt = res_df.tt.dropna().unique()[0]
        ranges = idxs[1:] - idxs[:-1]
        intervals = ranges // (num_extra_samples + 1)
        intervals = intervals[np.newaxis, :].T

        mul_range = np.arange(1, num_extra_samples + 1)[np.newaxis, :]
        samples_jumps = np.tile(mul_range, (len(ranges), 1))
        new_idxs = idxs[:-1, np.newaxis]
        take_idxs = (samples_jumps * intervals) + new_idxs
        take_idxs = np.concatenate([take_idxs.ravel(), idxs])
        final_df = res_df.iloc[take_idxs].sort_values(['visit_age_mo']).reset_index(drop=True)
        return final_df

    def draw_interp_results(self, df):

        data = df.iloc[:, :self.get_meta_idx()]
        orig_data_full = df[df.source == 'data']
        orig_data = orig_data_full.iloc[:, :self.get_meta_idx()]
        sort_arg = np.argsort(data.sum(axis=0))
        data: pd.DataFrame = data.iloc[:, sort_arg[::-1]].reset_index(drop=True)
        orig_data: pd.DataFrame = orig_data.iloc[:, sort_arg[::-1]].reset_index(drop=True)
        data_np = data.to_numpy()
        orig_data_np = orig_data.to_numpy()
        plot = plt.plot(df.visit_age_mo, data_np[:, 0], 'ro-')
        plt.show()
        plot = plt.plot(orig_data_full.visit_age_mo, orig_data_np[:, 0], 'bo-')
        plt.show()
        plot = plt.plot(df.visit_age_mo, data_np[:, 0], 'ro-', orig_data_full.visit_age_mo, orig_data_np[:, 0], 'b-')
        plt.show()
        print()

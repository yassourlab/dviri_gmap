import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 320)
from projects.functions import *

SAMPLE_TIME_SORTED = ['initial', 'twoweek', 'onemonth', 'twomonth', 'fourmonth', 'sixmonth', 'ninemonth', 'oneyear',
                      'sick']


class Classification:
    def __init__(self, norm_l7_path):
        self.merge_df, self.meta_idx = get_gmap_data(norm_l7_path)
        norm_merge_df, norm_meta_idx = get_gmap_data(norm_l7_path)
        best_features = [420, 260, 69, 264, 169, 269, 240, 304, 274, 17, 85, 246, 249, 314, 220, 253]
        filtered_meta_idx = len(best_features)

        # best_f_df = norm_merge_df.iloc[:,best_features].reset_index(drop=True)
        best_f_df = self.merge_df.iloc[:, best_features].reset_index(drop=True)
        best_f_df = best_f_df.div(best_f_df.sum(axis=1), axis=0).fillna(0)  # normalize to 1
        # best_f_df = best_f_df.loc[~(best_f_df.sum(axis=1) == 0)]
        # filtered_merge_df = filtered_merge_df.fillna(0)

        meta_df = norm_merge_df.iloc[:, norm_meta_idx:].reset_index(drop=True)
        filtered_merge_df = pd.concat([best_f_df, meta_df], axis=1)

        self.best_params = {'random_state': 666,
                            'n_estimators': 1200,
                            'min_samples_split': 20,
                            'min_samples_leaf': 2,
                            'max_features': 'sqrt',
                            'max_depth': None,
                            'class_weight': {True: 5, False: 1},
                            'bootstrap': False}

        self.filtered_params = {'random_state': 666, 'n_estimators': 1200, 'min_samples_split': 5,
                                'min_samples_leaf': 10, 'max_features': 'auto', 'max_depth': None, 'bootstrap': False,
                                'class_weight': {True: 4, False: 1}}
        self.filtered_merge_df = filtered_merge_df
        self.filtered_meta_idx = filtered_meta_idx

    def calc_is_sick(self, merge_df, filtered_merge_df, meta_idx, filtered_meta_idx, best_params, filtered_params,
                     verbose=True, train_ratio=0.7):
        num_runs = 1
        train_ratio = train_ratio
        merge_df_tt = get_merged_tt_df(merge_df, split_control=False, train_ratio=train_ratio)
        filtered_merge_df = filtered_merge_df.assign(tt=merge_df_tt.tt)

        filtered_clf = RandomForestClassifier(**filtered_params)

        filtered_merge_df = filtered_merge_df.loc[~pd.isna(filtered_merge_df.symptoms)]
        filtered_merge_df = filtered_merge_df.assign(is_sick=(filtered_merge_df.symptoms == 'Symptomatic'))
        f_is_sick_acc = get_avg_accuracy(filtered_clf, filtered_merge_df, filtered_meta_idx, num_runs=num_runs,
                                         y='is_sick',
                                         train_ratio=train_ratio, verbose=verbose)

        # On all features
        clf = RandomForestClassifier(**best_params)
        merge_df = merge_df.assign(is_sick=(merge_df.symptoms == 'Symptomatic'))
        merge_df = merge_df.loc[~pd.isna(merge_df.symptoms)]
        is_sick_acc = get_avg_accuracy(clf, merge_df, meta_idx, num_runs=num_runs, y='is_sick', train_ratio=train_ratio,
                                       verbose=verbose)

        if verbose:
            print("\nRandom Accuracy")
            filtered_merge_df['is_sick_str'] = filtered_merge_df.is_sick.astype(str)
            get_avg_classification_random_accuracy(filtered_merge_df, num_runs, label_name='is_sick_str',
                                                   train_ratio=train_ratio, verbose=verbose)

        return filtered_merge_df, merge_df, clf, filtered_clf

    def get_cls_prediction(self, clf, df, meta_idx, y_label, tt='test'):
        test_df = df[df.tt == tt]
        # test_df['orig_label'] = test_df.kmeans_label.copy()

        X_test = test_df.iloc[:, :meta_idx].values
        y_test = test_df[y_label].values
        test_pred = clf.predict(X_test)
        pred_proba = clf.predict_proba(X_test)
        max_proba = np.max(pred_proba, axis=1)
        test_df['max_proba'] = max_proba
        test_df['pred'] = test_pred
        #     test_df['y'] = y_test
        #     test_df['X'] = X_test
        return test_df, X_test, y_test

    def get_avg_tables(self, n_runs, merge_df, filtered_merge_df, meta_idx, filtered_meta_idx, best_params,
                       filtered_params, *args, **kwargs):
        filtered_merge_df, merge_df, clf, filtered_clf = self.calc_is_sick(merge_df, filtered_merge_df, meta_idx,
                                                                           filtered_meta_idx, best_params,
                                                                           filtered_params, verbose=False)

        precentage_list = list()
        count_list = list()
        filt_precentage_list = list()
        filt_count_list = list()
        for i in range(n_runs):
            precentage_np, count_np = self.get_labeled_predicion(clf=clf, merge_df=merge_df, **kwargs,
                                                                 return_tables=True,
                                                                 verbose=False)
            precentage_list.append(precentage_np)
            count_list.append(count_np)

            precentage_np, count_np = self.get_labeled_predicion(clf=filtered_clf, merge_df=filtered_merge_df, **kwargs,
                                                                 return_tables=True, verbose=False)
            filt_precentage_list.append(precentage_np)
            filt_count_list.append(count_np)

        precentage_avg = np.average(np.stack(precentage_list))
        count_avg = np.average(np.stack(count_list))

        filt_precentage_avg = np.average(np.stack(filt_precentage_list))
        filt_count_avg = np.average(np.stack(filt_count_list))
        return precentage_avg, count_avg, filt_precentage_avg, filt_count_avg

    def get_labeled_predicion(self, clf, merge_df, meta_idx, y_label, pivot_label, tt='test', norm_axis=1, round_n=3,
                              name='',
                              return_tables=False, verbose=True):
        test_df, X_test, y_test = get_cls_prediction(
            clf, merge_df, meta_idx, y_label=y_label, tt=tt)
        pred = test_df.pred.values
        label = test_df[pivot_label]

        two_cls_pred = [str(p) for p in pred]
        two_cls_y_test = [str(p) for p in label]
        #     print(f"Got test accuracy {accuracy_score(two_cls_y_test, two_cls_pred)}")

        # test_df = pd.DataFrame().assign(pred=pred,bin_pred=two_cls_pred,bin_label=two_cls_y_test) #type: pd.DataFrame
        pred_df = pd.DataFrame().assign(pred=pred, bin_pred=two_cls_pred,
                                        bin_label=two_cls_y_test, label=y_test)  # type: pd.DataFrame
        t = pred_df.melt(id_vars=['bin_label', 'label'],
                         value_vars=['pred'], value_name='predict')
        tbl = t.groupby(['bin_label', 'predict'])[
            'variable'].agg('count').unstack().fillna(0)

        percentage_tbl = tbl.div(tbl.sum(axis=norm_axis), axis=1 - norm_axis)
        precentage_np = percentage_tbl.to_numpy().round(round_n)
        count_np = tbl.to_numpy().round(round_n)

        if not verbose:
            if return_tables:
                return precentage_np, count_np
            else:
                return test_df

        print(percentage_tbl)
        print("\n\n")
        print(tbl.astype(int))

        self.draw_acc_count_heatmap(count_np, name, precentage_np, tbl)

        return test_df

    def draw_acc_count_heatmap(self, count_np, name, precentage_np, tbl):
        c = 'brwnyl'
        x = tbl.columns.astype(str).tolist()
        y = tbl.index.astype(str).tolist()
        fig_base = make_subplots(rows=2, cols=1)
        fig_percent = ff.create_annotated_heatmap(
            precentage_np, x=x, y=y, colorscale=c, zmin=0.5, zmax=1.0)
        fig_percent['layout']['xaxis']['side'] = 'top'
        fig_base.append_trace(fig_percent['data'][0], 1, 1)
        fig_cnt = ff.create_annotated_heatmap(precentage_np, annotation_text=count_np, x=x,
                                              y=y, colorscale=fig_percent.data[0]['colorscale'], zmin=0.5, zmax=1,
                                              zauto=False)
        fig_cnt['layout']['xaxis']['side'] = 'top'
        fig_base.append_trace(fig_cnt['data'][0], 2, 1)
        # Add annotation to the plot
        annot1 = list(fig_percent.layout.annotations)
        annot2 = list(fig_cnt.layout.annotations)
        for k in range(len(annot2)):
            annot2[k]['xref'] = 'x2'
            annot2[k]['yref'] = 'y2'
        fig_base.update_layout(annotations=annot1 + annot2)
        fig_base['layout']['xaxis']['side'] = 'top'
        fig_base.update_layout(
            title_text=f'{name} Model Prediction - Symptoms(rows)/is_sick(cols)', font=dict(size=18))
        fig_base.show()

    filtered_params = {'random_state': 666,
                       'n_estimators': 1200,
                       'min_samples_split': 5,
                       'min_samples_leaf': 10,
                       'max_features': 'auto',
                       'max_depth': None,
                       'bootstrap': False}

    best_params = {'random_state': 666,
                   'n_estimators': 1200,
                   'min_samples_split': 20,
                   'min_samples_leaf': 2,
                   'max_features': 'sqrt',
                   'max_depth': None,
                   'bootstrap': False}
    # norm_merge_df,norm_meta_idx = get_gmap_data(l7_path)

    # norm_merge_df, norm_meta_idx = get_gmap_data(norm_l7_path)


def main():
    # clf, merge_df, meta_idx, y_label, pivot_label
    classification = Classification(norm_l7_path)
    merge_df, meta_idx = classification.merge_df, classification.meta_idx
    filtered_merge_df = classification.filtered_merge_df
    filtered_meta_idx = classification.filtered_meta_idx
    best_params = classification.best_params
    filtered_params = classification.filtered_params

    precentage_avg, count_avg, filt_precentage_avg, filt_count_avg = \
        classification.get_avg_tables(2, merge_df, filtered_merge_df, best_params, filtered_params, meta_idx,
                                      filtered_meta_idx, y_label='is_sick',
                                      pivot_label='symptoms', norm_axis=1, round_n=2,
                                      name='Filtered')
    pass


if __name__ == '__main__':
    main()

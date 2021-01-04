import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def split_is_train(x,th=0.75):
    val = np.random.rand()
    is_train = val<th
    x['is_train'] = is_train
    return x


def build_trainig_data(merge_df,meta_idx,test_size=0.25,split_by_control=False):
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

        ap_data = ap_data.groupby(['record_id']).apply(lambda x: split_is_train(x, 1 - test_size))
        X_control_train = ap_data.loc[ap_data.is_train].iloc[:, :meta_idx]
        y_control_train = ap_data.loc[ap_data.is_train].iloc[:, meta_idx:].loc[:,'visit_age_mo'].values

        X_control_test = ap_data.drop(X_control_train.index).iloc[:,:meta_idx]
        y_control_test = ap_data.drop(X_control_train.index).iloc[:, :meta_idx].loc[:,'visit_age_mo'].values




        return {"X_train": X_train, "X_test": X_test, "y_train": y_train, 'y_test':y_test,
                "X_control_train": X_control_train,
                "y_control_train": y_control_train,
                "X_control_test": X_control_test,
                "y_control_test": y_control_test,
                }
    else:
        ap_data = merge_df
        X = merge_df.iloc[:, :meta_idx].values
        meta = merge_df.iloc[:, meta_idx:]
        y = meta.loc[:, 'visit_age_mo'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=666)

        return X_train,X_test,y_train,y_test
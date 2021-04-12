# from astunparse.unparser import main
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model

def get_gmap_data(*args, **kwargs):
    """
    return merge_df,meta_idx
    """
    data_path = "./dviri_gmap/projects/py/nb/data/feature-table-norm-l7.csv"
    otu_data = pd.read_csv(data_path,sep='\t',index_col=['OTU ID'])
    examples_data = otu_data.T
    metadata = pd.read_csv("./dviri_gmap/projects/py/nb/data/edited_metadata.csv",sep='\t')
    merge_df = examples_data.merge(metadata,left_index=True,right_on=['sampleID'])
    merge_df = merge_df.assign(is_control = (merge_df.symptoms == 'Control'))
    meta_idx = examples_data.shape[1]
    return merge_df,meta_idx

def get_is_sick_dataset(merged_df, data_idx, batch_size=32, binary_data=False):
    control_merged_data = merged_df[merged_df.symptoms == "Control"]
    data_df = control_merged_data.iloc[:, data_idx:]

    data_df = data_df.loc[~pd.isna(data_df.symptoms)]
    data_df = data_df.assign(label = (data_df.symptoms == 'Symptomatic'))

    test_df = data_df.sample(frac=0.2, random_state=666)
    train_df = data_df.drop(test_df.index)

    train_ds = data_to_ds(batch_size, binary_data, train_df)
    test_ds = data_to_ds(-1, binary_data, test_df)
    return train_ds, test_ds

def data_to_ds(batch_size: int, binary_data: bool, df: pd.DataFrame):
    """
    Given dataframe of examples, will return that df as tensorflow dataset
    Args:
        batch_size: size of each batch
        binary_data: if True will make data to be binary (instead of quantity it will becore 0 or 1
        df:

    Returns:

    """
    dtype = np.bool if binary_data else np.float32
    X = df.values
    X = X.astype(dtype)
    Y = df.label.values.astype(np.float32)
    Y = Y[..., np.newaxis]
    if batch_size > 0:
        train_ds = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    else:
        train_ds = tf.data.Dataset.from_tensor_slices((X, Y))
    return train_ds
    
def main():
    merge_df,meta_idx = get_gmap_data()
    train_ds, test_ds = get_is_sick_dataset(merge_df,meta_idx, batch_size = 32)
    model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(meta_idx)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.fit(train_ds, epoches=10)
    pass



    if __name__ == "__main__":
        main()
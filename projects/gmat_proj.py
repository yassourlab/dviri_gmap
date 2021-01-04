# from astunparse.unparser import main
import os
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.models import Model


class CustomModel(Model):

    def __init__(self, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.dense1 = Dense(64, activation='relu', name="Dense_1")
        self.dropout1 = Dropout(0.2)
        self.dense2 = Dense(32, activation='relu', name="Dense_2")
        self.dropout2 = Dropout(0.2)
        self.dense3 = Dense(16, activation='relu', name="Dense_3")
        self.dropout3 = Dropout(0.2)
        self.predict_layer = Dense(1)

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        # x = self.dropout1(x)
        x = self.dense2(x)
        # x = self.dropout2(x)
        x = self.dense3(x)
        # x = self.dropout3(x)
        x = self.predict_layer(x)
        return x

def dataframe_to_dataset(data_df):
    dataframe = data_df.copy()
    labels = dataframe.pop("label")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def pred_dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe)))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def setup_data(normalize_val=0):
    data_path = "./data/starting_data/not-filtered-data/"
    features_path = os.path.join(data_path, "OTU_FIX_feature-table-l7.csv")
    edited_metadta = f"{data_path}/../edited_metadata.tsv"
    data_df = pd.read_csv(features_path, sep='\t')

    # if drop_unassigned:
    data_df = data_df.loc[~data_df["OTU ID"]
        .str.contains("Unassigned")].reset_index(drop=True)

    # transform on groupby perform the action but keep all rows and just duplicate the values to match original DF
    data_df_summed = data_df.groupby(['OTU ID']).transform("sum")
    # gropuby removes the column on wihch it was used. So I'm copying it from the original data_df
    data_df_summed['OTU ID'] = data_df['OTU ID']
    # reorder the columns such the "OTU ID" is first column
    data_df_summed = data_df_summed[['OTU ID'] + [c for c in data_df_summed if c not in ['OTU ID']]]

    # Drop the duplicates rows now with same values thanks to the transform method
    data_df_summed.drop_duplicates(inplace=True)

    meta_df = pd.read_csv(edited_metadta, sep='\t').rename(columns={"sampleID": "sample_name"})

    # replace the "sample_time" with numbers corresponding the time
    sample_time_enum = {sample_time: i for i, sample_time in enumerate(
        meta_df.sort_values('visit_age_mo').sample_time.unique().tolist())}
    sample_time_enum['sick'] = len(sample_time_enum) + 5

    data_df_indexed = data_df_summed.set_index("OTU ID", drop=True)
    meta_df = meta_df.assign(sample_time_enum=meta_df.sample_time)
    meta_df.replace({"sample_time_enum": sample_time_enum}, inplace=True)
    meta_df = meta_df[['sample_time_enum'] + [c for c in meta_df if c not in ['sample_time_enum']]]
    meta_df.head()

    # Set small values to 0
    data_df_indexed[data_df_indexed < normalize_val] = 0

    data_idx = meta_df.shape[1]
    merged_df = meta_df.merge(data_df_indexed.T, right_index=True, left_on=['sample_name'])

    return merged_df, data_idx





def get_training_dataset(merged_df, data_idx, batch_size=32, binary_data=False):
    control_merged_data = merged_df[merged_df.symptoms == "Control"]
    data_df = control_merged_data.iloc[:, data_idx:]
    data_df['label'] = control_merged_data.visit_age_mo

    test_df = data_df.sample(frac=0.2, random_state=666)
    train_df = data_df.drop(test_df.index)

    train_ds = data_to_ds(batch_size, binary_data, train_df)
    test_ds = data_to_ds(-1, binary_data, test_df)
    return train_ds, test_ds
    # Y_data = control_merged_data.visit_age_mo.values.astype(np.float32)


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


def loss_function(labels, prediction, margin=0.):
    loss = tf.maximum(tf.abs(prediction - labels) - margin, 0)
    loss = tf.reduce_mean(loss)
    # print(prediction[0].numpy()[0], labels[0].numpy()[0])
    return loss


class GmatAcc:
    def __init__(self, margin) -> None:
        self.margin = margin
        self.acc = 0
        self.n = 0

    def __call__(self, labels, prediction):
        self.calc_accuracy(labels, prediction)

    def calc_accuracy(self, labels, prediction):
        loss = np.abs(prediction - labels)
        succ = np.count_nonzero(loss < self.margin) / len(loss)
        # succ = tf.math.count_nonzero(tf.less(loss, self.margin))
        self.acc += succ
        self.n += 1
        return self.acc

    def result(self):
        return self.acc/self.n

    def reset_states(self):
        self.acc = 0
        self.n = 0


def build_test(model: tf.keras.Model, loss_object: tf.losses.Loss, margin=0.1):
    compute_loss = tf.keras.metrics.Mean(name='train_loss')
    gmat_acc = GmatAcc(margin)

    def test_step(data, labels, norm_factor=1e-3):
        prediction = model(data)
        loss = loss_function(labels, prediction, margin=margin)
        loss = compute_loss(loss)
        acc = gmat_acc(labels, prediction)
        return loss, prediction, acc

    return test_step, compute_loss, gmat_acc


def build_train(model: tf.keras.Model, loss_object: tf.losses.Loss, learning_rate=0.001, margin=0.1):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    compute_loss = tf.keras.metrics.Mean(name='train_loss')
    gmat_acc = GmatAcc(margin)

    def train_step(data, labels, norm_factor=1e-3):
        with tf.GradientTape() as tape:
            prediction = model(data)
            loss = loss_function(labels, prediction, margin=margin)
            gradients = tape.gradient(loss, model.trainable_variables)

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = compute_loss(loss)
        acc = gmat_acc(labels, prediction)
        return loss, prediction, acc

    return train_step, compute_loss, gmat_acc


EPOCHS = 100000


def run_model(merged_df, data_idx):
    date_time_obj = datetime.now()
    ts = date_time_obj.strftime("%Y_%m_%d")
    train_ds, test_ds = get_training_dataset(merged_df, data_idx, batch_size=32, binary_data=True)

    model = CustomModel(name='customModel')
    train_step, train_loss, train_acc = build_train(model, loss_function, 0.001)
    test_step, test_loss, test_acc = build_test(model, loss_function, 0.01)
    step = 0
    num_samples = 0

    train_writer = tf.summary.create_file_writer(f"./logs/{ts}/train/")
    test_writer = tf.summary.create_file_writer(f"./logs/{ts}/test/")
    for epoch in range(EPOCHS):
        for data, labels in train_ds:
            train_step(data, labels)
            step += 1

            if step % 100 == 0:
                print(f"Loss: {train_loss.result()}, acc: {train_acc.result()}")
                tf.summary.scalar("loss", train_loss.result(), step=step)
                tf.summary.scalar("acc", train_acc.result(), step=step)
                with train_writer.as_default():
                    train_loss.reset_states()
                    train_acc.reset_states()

        if epoch % 3 == 0 and epoch != 0:

            for data, labels in test_ds:
                data = data[tf.newaxis,:]  # make data shape to be (1,N) instead of (N)
                test_step(data, labels)


                print(f"Loss: {test_loss.result()}, acc: {test_acc.result()}")
                tf.summary.scalar("loss", test_loss.result(), step=step)
                tf.summary.scalar("acc", test_acc.result(), step=step)
                with test_writer.as_default():
                    test_loss.reset_states()
                    test_acc.reset_states()
                break
    pass
    # model.compile(optimizer='Adam',
    #           loss=tf.keras.losses.MeanSquaredError(),
    #           metrics=['accuracy'])

    # history = model.fit(X,Y,
    #                 batch_size=8,
    #                 epochs= 20)

    pass


def main():
    merged_df, data_idx = setup_data(normalize_val=10)
    run_model(merged_df, data_idx)


if __name__ == "__main__":
    main()

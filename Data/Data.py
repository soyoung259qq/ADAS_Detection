import tensorflow as tf
from Utils.Label import LabelEncoder
import tensorflow_datasets as tfds
from Data.Preprocessing import preprocess_data
import os
import zipfile


def prepare_data():
    """
    ## Downloading the COCO2017 dataset

    Training on the entire COCO2017 dataset which has around 118k images takes a
    lot of time, hence we will be using a smaller subset of ~500 images for
    training in this example.
    """

    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    tf.keras.utils.get_file(filename, url)

    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("./")


def get_label_info():
    # set `data_dir=None` to load the complete dataset
    (_, _), dataset_info = tfds.load(
        "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
    )

    return dataset_info.features["objects"]["label"].int2str

def get_train_data(batch_size=2):

    prepare_data()

    """
    ## Load the COCO2017 dataset using TensorFlow Datasets
    """
    label_encoder = LabelEncoder()
    # set `data_dir=None` to load the complete dataset
    (train_dataset, val_dataset), dataset_info = tfds.load(
        "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
    )
    """
    ## Setting up a `tf.data` pipeline

    To ensure that the model is fed with data efficiently we will be using
    `tf.data` API to create our input pipeline. The input pipeline
    consists for the following major processing steps:

    - Apply the preprocessing function to the samples
    - Create batches with fixed batch size. Since images in the batch can
    have different dimensions, and can also have different number of
    objects, we use `padded_batch` to the add the necessary padding to create
    rectangular tensors
    - Create targets for each sample in the batch using `LabelEncoder`
    """

    autotune = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    return train_dataset, val_dataset

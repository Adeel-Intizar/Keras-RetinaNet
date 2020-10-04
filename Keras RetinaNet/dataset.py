import os 
import zipfile
import tensorflow as tf
import tensorflow_datasets as tfds
from utils import preprocess_data
from label_encoder import LabelEncoder


def data():
    
    url = "https://github.com/srihari-humbarwadi/datasets/releases/download/v0.1.0/data.zip"
    filename = os.path.join(os.getcwd(), "data.zip")
    tf.keras.utils.get_file(filename, url)

    with zipfile.ZipFile("data.zip", "r") as z_fp:
        z_fp.extractall("./")

    #  set `data_dir=None` to load the complete dataset
    (train_dataset, val_dataset), dataset_info = tfds.load(
        "coco/2017", split=["train", "validation"], with_info=True, data_dir="data")
    
    batch_size = 2
    autotune = tf.data.experimental.AUTOTUNE
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
    train_dataset = train_dataset.map(
        LabelEncoder.encode_batch, num_parallel_calls=autotune)
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True)
    val_dataset = val_dataset.map(LabelEncoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)
    
    return train_dataset, val_dataset, dataset_info


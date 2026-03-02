#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This is a script to evaluate the performance of a trained keras model.

__author__     = "Christian Scholz and Sandy Scholz"
__copyright__  = "Copyright 2021, authors"
__credits__    = ["Christian Scholz", "Sandy Scholz"]
__license__    = "GPLv3"
__version__    = "1.0"
__maintainer__ = "Christian Scholz and Sandy Scholz"
__email__      = "coscholz1984@gmail.com"
__status__     = "Development"
__summary__    = "Here we evaluate the performance of a convolutional neural network"

Usage
--------
python CNN_Evaluate.py [model path]
"""

# %% Dependencies
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import GStools as gst

# %% Global constants
NORMALIZE_DATA = False

# %% Font settings
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16

# %% Model selection
if len(sys.argv) == 1:
    IMPORT_PATH_KERAS = "model_CNN_2D"
    print("No input given, using 'model_CNN_2D'")
elif len(sys.argv) == 2:
    IMPORT_PATH_KERAS = sys.argv[1]
    print(f"Model {IMPORT_PATH_KERAS} used")
else:
    raise NameError("Too many inputs given.")

# %% Load model and history
model = tf.keras.models.load_model(IMPORT_PATH_KERAS)
history = pickle.load(open(f"./{IMPORT_PATH_KERAS}/trainHistoryDict", "rb"))

# %% Load datasets
FILEPREFIX = "Dataset_3D_" if IMPORT_PATH_KERAS == "model_CNN_3D" else "Dataset_2D_"

if IMPORT_PATH_KERAS == "model_CNN_2D_2nd":
    dataset_train = np.array(pickle.load(open(FILEPREFIX + "train2.p", "rb")))
    dataset_val = np.array(pickle.load(open(FILEPREFIX + "val2.p", "rb")))
    labels_train = pickle.load(open(FILEPREFIX + "train2_label.p", "rb"))
    labels_val = pickle.load(open(FILEPREFIX + "val2_label.p", "rb"))
else:
    dataset_train = np.array(pickle.load(open(FILEPREFIX + "train.p", "rb")))
    dataset_val = np.array(pickle.load(open(FILEPREFIX + "val.p", "rb")))
    labels_train = pickle.load(open(FILEPREFIX + "train_label.p", "rb"))
    labels_val = pickle.load(open(FILEPREFIX + "val_label.p", "rb"))

dataset_test = np.array(pickle.load(open(FILEPREFIX + "test.p", "rb")))
labels_test = pickle.load(open(FILEPREFIX + "test_label.p", "rb"))

# %% Normalize data if required
if NORMALIZE_DATA:
    dataset_train = tf.keras.utils.normalize(dataset_train, axis=-1)
    dataset_val = tf.keras.utils.normalize(dataset_val, axis=-1)
    dataset_test = tf.keras.utils.normalize(dataset_test, axis=-1)

# %% Add channel dimension for 3D model
if IMPORT_PATH_KERAS == "model_CNN_3D":
    dataset_train = np.expand_dims(dataset_train, axis=4)
    dataset_val = np.expand_dims(dataset_val, axis=4)
    dataset_test = np.expand_dims(dataset_test, axis=4)

# %% Convert labels to integer indices
labels_train_int = gst.intLabels(labels_train)
labels_val_int = gst.intLabels(labels_val)
labels_test_int = gst.intLabels(labels_test)

# %% Handle special test-label remapping
if IMPORT_PATH_KERAS == "model_CNN_2D_2nd":
    labels_test_int = np.where(labels_test_int == 14, 15, labels_test_int)
    labels_test_int = np.where(labels_test_int == 13, 14, labels_test_int)

ticks_test = gst.get_greek_labels(labels_test_int.max() + 1)

# %% Plot training history
plt.figure()
plt.plot(history["sparse_categorical_accuracy"])
plt.plot(history["val_sparse_categorical_accuracy"])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(["train", "val"], loc="lower right")
plt.tight_layout()
plt.show()

# %% Predictions
labels_train_pred = model.predict(dataset_train).argmax(axis=1)
labels_val_pred = model.predict(dataset_val).argmax(axis=1)
labels_test_pred = model.predict(dataset_test).argmax(axis=1)

# %% Confusion matrices
cm_train = confusion_matrix(labels_train_int, labels_train_pred)
cm_val = confusion_matrix(labels_val_int, labels_val_pred)
cm_test = confusion_matrix(labels_test_int, labels_test_pred)

# %% Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
titles = ["Training", "Validation", "Test"]
cms = [cm_train, cm_val, cm_test]

for ax, cm, title, split in zip(
    axes,
    cms,
    titles,
    ["train", "val", "test"]
):
    n_classes = cm.shape[0]

    if split == "test":
        labels = ticks_test
    else:
        labels = gst.get_greek_labels(n_classes)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        xticks_rotation="horizontal",
        include_values=True
    )

    ax.set_title(title)

plt.tight_layout()
plt.show()

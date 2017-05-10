#Import modules
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sbs

from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
from subprocess import check_output

labels_df = pd.read_csv('label_names.csv')
filenames = ["video_level/traina{}.tfrecord".format(i) for i in range(10)]
print("we have {} unique labels in the dataset".format(len(labels_df['label_name'].unique())))

labels = []
textual_labels = []
textual_labels_nested = []
total_sample_counter = 0

label_counts = []

for filename in filenames:
    for example in tf.python_io.tf_record_iterator(filename):
        total_sample_counter += 1
        tf_example = tf.train.Example.FromString(example)
        # print(tf_example)
        label_example = list(tf_example.features.feature['labels'].int64_list.value)
        label_counts.append(len(label_example))
        labels = labels + label_example
        label_example_textual = list(labels_df[labels_df['label_id'].isin(label_example)]['label_name'])
        textual_labels_nested.append(set(label_example_textual))
        textual_labels = textual_labels + label_example_textual
        textual_labels = textual_labels + label_example_textual
        if len(label_example_textual) != len(label_example):
            print('label names lookup failed: {} vs {}'.format(label_example, label_example_textual))




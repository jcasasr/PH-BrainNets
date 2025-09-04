from Subject import Subject
from config import *
import utils
import os
import numpy as np


# load data
subjects = utils.load_subjects(path_input=PATH, subject_pattern=SUBJECT_PATTERN, debug=DEBUG)

subject = subjects[0]

# check all available attributes
print("Attributes:")
for key in subject._attributes.keys():
    value = subject.get_attribute(key)
    if value is not None:
        print("   {} [type: {}, value: {}]".format(key, type(value), value))
    else:
        # if the attribute is None, we can still print its name
        # but we don't know its type or value
        print("   {}".format(key))

# check all available matrices
print("Matrices  :")
for key in subject._matrices.keys():
    value = subject.get_matrix(key)
    if value is not None:
        print("   {} [shape: {}, range: {:.4f} - {:.4f}]".format(key, value.shape, np.min(value), np.max(value)))
    else:
        # if the matrix is None, we can still print its name
        # but we don't know its shape or range
        print("   {}".format(key))

# check all available metrics
print("Metrics   :")
for key in subject._metrics.keys():
    value = subject.get_metric(key)
    if value is not None:
        print("   {} [shape: {}, range: {:.4f} - {:.4f}]".format(key, value.shape, np.min(value), np.max(value)))
    else:
        # if the metric is None, we can still print its name
        # but we don't know its shape or range
        print("   {}".format(key))

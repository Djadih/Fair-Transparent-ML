import sys, os
import numpy as np
sys.path.append("../")
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, GermanDataset, CompasDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.metrics.utils import compute_boolean_conditioning_vector

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas, load_preproc_data_german

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.metrics import accuracy_score

from IPython.display import Markdown, display
import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def get_plain_and_debaised_model_adversarial_debiasing():
    # Block printing to not distract the subject.
    # sys.stdout = open(os.devnull, 'w')
    ## trains on whole dataset
    dataset_orig= load_preproc_data_adult()
    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    sess_plain = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='plain_classifier',
                          debias=False,
                          sess=sess_plain)

    plain_model.fit(dataset_orig_train)

    tf.reset_default_graph()
    sess_debiased = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess_debiased)

    debiased_model.fit(dataset_orig_train)
    sys.stdout = sys.__stdout__

    return plain_model, debiased_model


def create_single_entry_adult_dataset(race, sex, age, education_years):
    # subject to change if we decide to/are able to use more features than those which appear in the example notebooks
    arr = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])

    arr[0][0] = race
    arr[0][1] = sex
    if age <= 19:
        arr[0][2] = 1.
    elif age <= 29:
        arr[0][3] = 1.
    elif age <= 39:
        arr[0][4] = 1.
    elif age <= 49:
        arr[0][5] = 1.
    elif age <= 59:
        arr[0][6] = 1.
    elif age <= 69:
        arr[0][7] = 1.
    else:
        arr[0][8] = 1.

    if education_years < 6:
        arr[0][16] = 1.
    elif education_years == 6:
        arr[0][9] = 1.
    elif education_years == 7:
        arr[0][10] = 1.
    elif education_years == 8:
        arr[0][11] = 1.
    elif education_years == 9:
        arr[0][12] = 1.
    elif education_years == 10:
        arr[0][13] = 1.
    elif education_years == 11:
        arr[0][14] = 1.
    elif education_years == 12:
        arr[0][15] = 1.
    else:
        arr[0][17] = 1.
    
    dataset_replaced_data = load_preproc_data_adult()
    dataset_replaced_data.features = arr
    dataset_replaced_data.age = age
    dataset_replaced_data.edu = education_years
    return dataset_replaced_data

def predict_income_adversarial_debiasing(model, user_input):
    pred = model.predict(user_input).labels[0][0]
    
    races = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    genders = ['Female', 'Male']

    if int(user_input.features[0][0]) == 0:
        race_print = "white"
    else:
        race_print = "non-white"
    if int(user_input.features[0][1]) == 0:
        sex_print = "female"
    else:
        sex_print = "male"

    if pred == 1.0:
        print(f"The model predicts that a {user_input.age} year old {race_print} {sex_print} with {user_input.edu} years of education DOES have an income greater than 50k.")
    elif pred == 0.0:
        print(f"The model predicts that a {user_input.age} year old {race_print} {sex_print} with {user_input.edu} years of education DOES NOT have an income greater than 50k.")
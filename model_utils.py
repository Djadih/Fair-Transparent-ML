import sys
import numpy as np
sys.path.append("../")

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult
from aif360.algorithms.inprocessing import GerryFairClassifier, ExponentiatedGradientReduction, AdversarialDebiasing
from sklearn.linear_model import LogisticRegression

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def plain_training(dataset, privileged_groups, unprivileged_groups):
    tf.reset_default_graph()
    sess_plain = tf.Session()
    plain_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                        unprivileged_groups = unprivileged_groups,
                        scope_name='plain_classifier',
                        debias=False,
                        sess=sess_plain)

    plain_model.fit(dataset)

    print("Plain model completed training!")

    return plain_model

def adversarial_debiasing(dataset, privileged_groups, unprivileged_groups):
    # train on entire dataset, not split
    # dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    tf.reset_default_graph()
    sess_debiased = tf.Session()
    debiased_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name='debiased_classifier',
                          debias=True,
                          sess=sess_debiased)

    debiased_model.fit(dataset)

    print("Adversarial model completed training!")

    return debiased_model

def exponentiated_gradient_reduction(dataset, constraints):
    # Exponentiated Gradient Reduction
    expgrad = ExponentiatedGradientReduction(constraints=constraints, estimator=LogisticRegression())
    expgrad.fit(dataset)

    print("Exponentiated Gradient Reduction model completed training!")

    return expgrad

def calibrated_eqodds_postprocessing(dataset_orig, dataset_orig_pred, privileged_groups, unprivileged_groups):
    # Odds equalizing post-processing algorithm
    cost_constraint = 'weighted'

    # Learn parameters to equalize odds and apply to create a new dataset
    cpp = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                        unprivileged_groups = unprivileged_groups,
                                        cost_constraint=cost_constraint)
    cpp = cpp.fit(dataset_orig, dataset_orig_pred)

    print("Calibrated Eq Odds Postprocessing model completed training!")

    return cpp

def gerry_fair_trained_model(dataset):
    gerry_fair_model = GerryFairClassifier(C=100, printflag=False, gamma=.005, fairness_def='FP',
             max_iters = 500, heatmapflag=False)

    gerry_fair_model.fit(dataset, early_termination=True)

    print("Gerry Fair model completed training!")

    return gerry_fair_model

def create_single_entry_adult_dataset(race, sex, age, education_years, model_type=None):
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
    if model_type == "Gerry_Fair":
        dataset_replaced_data.protected_attributes = [race,sex]
        dataset_replaced_data.instance_names = ['0']
        dataset_replaced_data.instance_weights = [0.5]
        dataset_replaced_data.labels = np.array([[0.0]])
    return dataset_replaced_data


def prompt_for_ranking():
    ret_rankings = []

    models = ["Plain", "Adversarial_Debiased", "Calibrated_Eq_Odds_Postprocessing", "Gerry_Fair"]
    model_idx = [0, 1, 2, 3]
    model_idx_names = ["0: Plain", "1: Adversarial", "2: Calibrated Odds", "3: Gerry_Fair"]
    
    valid_input = False
    while not valid_input:
        input_str = "Which model was the most fair? "
        for model_name in model_idx_names:
            if model_name != " ":
                input_str += model_name + "  "
        try:
            input_ = int(input(input_str))
        except Exception:
            input_ = -1
        if input_ not in model_idx:
            print("Try again")
        else:
            valid_input = True
            ret_rankings.append(models[input_])
            model_idx_names[input_] = " "
            model_idx.remove(input_)

    valid_input = False
    while not valid_input:
        input_str = "Which model was the next most fair? "
        for model_name in model_idx_names:
            if model_name != " ":
                input_str += model_name + "  "
        try:
            input_ = int(input(input_str))
        except Exception:
            input_ = -1
        if input_ not in model_idx:
            print("Try again")
        else:
            valid_input = True
            ret_rankings.append(models[input_])
            model_idx_names[input_] = " "
            model_idx.remove(input_)

    valid_input = False
    while not valid_input:
        input_str = "Which model was the next most fair? "
        for model_name in model_idx_names:
            if model_name != " ":
                input_str += model_name + "  "
        try:
            input_ = int(input(input_str))
        except Exception:
            input_ = -1
        if input_ not in model_idx:
            print("Try again")
        else:
            valid_input = True
            ret_rankings.append(models[input_])
            model_idx_names[input_] = " "
            model_idx.remove(input_)

    

    ret_rankings.append(models[model_idx[0]])
    return ret_rankings



def predict_income_adversarial_debiasing(model, user_input):
    # could use a touch up, maybe including the features we want them to think are being considered or to change non-white to be whatever they actually wanted it to represent.
    pred = model.predict(user_input).labels[0][0]
    
    races = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    genders = ['Female', 'Male']

    if int(user_input.features[0][0]) == 1:
        race_print = "white"
    else:
        race_print = "non-white"
    if int(user_input.features[0][1]) == 0:
        sex_print = "female"
    else:
        sex_print = "male"

    # print(f"The model predicts that a {user_input.age} year old {race_print} {sex_print} with {user_input.edu} years of education has a {round(pred*100, 2)}% chance of having an income greater than 50k.")
    if pred == 1.0:
        print(f"The model predicts that a {user_input.age} year old {race_print} {sex_print} with {user_input.edu} years of education makes MORE than 50k.")
    elif pred == 0.0:
        print(f"The model predicts that a {user_input.age} year old {race_print} {sex_print} with {user_input.edu} years of education makes LESS than 50k.")
    return pred


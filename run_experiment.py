from model_utils import *
from survey_subject import SurveySubject, Query
import concurrent.futures

def get_subject_info():
    print("--------------------")
    print("Beginning survey session... Training models in the background...")
    print("--------------------")

    subject_name = input("Please enter the subject's name/unique identifer: ")
    subject_age = input("Please enter the subject's age: ")
    subject_gender = input("Please enter the subject's gender: ")
    subject_race = input("Please enter the subject's race: ")

    print("Thank you. Beginning survey as soon as models complete training...")
    return SurveySubject(subject_name, subject_age, subject_gender, subject_race)

def adversarial_debiasing_query(subject, model_list):
    # runs through the process of gathering user input to select the model, select the features, 
    # and returning the output while logging the choices 
    query_log = Query()

    model_names = ["Plain", "Adversarial_Debiased", "Calibrated_Eq_Odds_Postprocessing"]
    
    model_choice = input("Which model should be used? [0: Plain, 1: Adversarial, 2: Calibrated Odds]: ")
    if model_choice == '-1':
        return False
    model = model_list[int(model_choice)]
    model_choice = model_names[int(model_choice)]

    sex_input = input("Indicate the person's gender: [m] or [f]: ")
    if sex_input == 'f':
        sex_input = 0.0
    elif sex_input == 'm':
        sex_input = 1.0
    race_input = input("Indicate the person's race: ")
    if race_input == "white":
        race_input = 0.0
    else:
        race_input = 1.0
    age_input = int(input("Indicate the person's age in years: "))
    education_input = int(input("Indicate the person's number of years in education: [typically 0-13]:\n"))

    query_input = create_single_entry_adult_dataset(race_input, sex_input, age_input, education_input)
    query_log.set_features(query_input.features)

    pred = predict_income_adversarial_debiasing(model, query_input)
    query_log.set_output(pred)

    subject.log_completed_query(query_log)

    return True

def run_experiment():
    # Implemented multi-threading so that the user can query the model while the model is training.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        subject_info_t = executor.submit(get_subject_info)

        privileged_groups = [{'sex': 1}]
        unprivileged_groups = [{'sex': 0}]
        
        dataset_orig= load_preproc_data_adult()

        # Need to train plain model first.
        plain_model = plain_training(dataset_orig, privileged_groups, unprivileged_groups)
        adversarial_model = adversarial_debiasing(dataset_orig, privileged_groups, unprivileged_groups)
        cpp_model = calibrated_eqodds_postprocessing(dataset_orig, plain_model.predict(dataset_orig), privileged_groups, unprivileged_groups)

        all_models = [plain_model, adversarial_model, cpp_model]
        print("Training completed!")

    subject = subject_info_t.result()
    # allow the user to input their own queries
    continue_session = True
    while continue_session:
        continue_session = adversarial_debiasing_query(subject, all_models)

    # After session completed, print all queries to see if the logging worked
    # in a real session we would save as a file probably
    subject.print_all_queries()    
    return

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_experiment()
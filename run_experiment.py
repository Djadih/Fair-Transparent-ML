from model_utils import *
from survey_subject import SurveySubject, Query

def adversarial_debiasing_query(subject, model_list):
    # runs through the process of gathering user input to select the model, select the features, 
    # and returning the output while logging the choices 
    query_log = Query()
    
    model_selected = False
    while not model_selected:
        model_choice = input("Which model should be used? plain [1] adversarially debaised [2]")
        if model_choice == '1':
            model = model_list[0]
            query_log.set_model_name("Adversarial_debiasing_plain")
            model_selected = True
        elif model_choice == '2':
            model = model_list[1]
            query_log.set_model_name("Adversarial_debiasing_debiased")
            model_selected = True
        elif model_choice == '-1':
            return False
        else:
            print("Invalid input. ", end="")

    sex_input = input("Indicate the person's gender: [m] or [f]\n")
    if sex_input == 'f':
        sex_input = 0.0
    elif sex_input == 'm':
        sex_input = 1.0
    race_input = input("Indicate the person's race:\n")
    if race_input == "white":
        race_input = 0.0
    else:
        race_input = 1.0
    age_input = int(input("Indicate the person's age in years:\n"))
    education_input = int(input("Indicate the person's number of years in education: [typically 0-13]:\n"))

    query_input = create_single_entry_adult_dataset(race_input, sex_input, age_input, education_input)
    query_log.set_features(query_input.features)

    pred = predict_income_adversarial_debiasing(model, query_input)
    query_log.set_output(pred)

    subject.log_completed_query(query_log)

    return True

def run_experiment():

    print("Beginning survey session... Please enter the subject's name/unique identifer:", end="")
    subject_name = input(" ")
    subject = SurveySubject(subject_name)

    print("Gathering models...")
    adversarial_plain_model, adversarial_debiased_model = get_plain_and_debaised_model_adversarial_debiasing()
    # similar functions to gether models using other debiasing methods
    all_models = [adversarial_plain_model, adversarial_debiased_model]

    # here we might run through some pre-selected examples
    # perhaps beginning with the plain model followed by the debiased one, or some other procedure

    # allow the user to input their own queries
    continue_session = True
    while continue_session:
        continue_session = adversarial_debiasing_query(subject, all_models)

    # After session completed, print all queries to see if the logging worked
    # in a real session we would save as a file probably
    subject.print_all_queries()    
    return

if __name__ == "__main__":
    run_experiment()
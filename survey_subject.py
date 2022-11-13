

class Query():
    def __init__(self, modelType=None, featureVector=None, model_output=None, featureNames=None):
        self.model = modelType
        self.features = featureVector
        self.output = model_output
        self.featureNames = featureNames
        return

    def set_model_name(self, modelName):
        self.model = modelName
    
    def set_features(self, featureVector):
        self.features = featureVector

    def set_output(self, modelOutput):
        self.output = modelOutput


class SurveySubject():
    def __init__(self, subjectName):
        self.subjectName = subjectName
        self.queries = []
        return

    def log_completed_query(self, query):
        self.queries.append(query)
        return

    def print_all_queries(self):
        # mostly for debugging purposes
        for query in self.queries:
            print(f"Model: {query.model}")
            print(f"Features: {query.features}")
            print(f"Output: {query.output}\n")

    def save_session_data(self):
        ## TODO: save the data gathered in the session 
        ## to some format like csv or other
        return

import pandas as pd

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

    def set_featureNames(self, featureNames):
        self.featureNames = featureNames


class SurveySubject():
    def __init__(self, subjectName, subjectAge, subjectGender, subjectRace):
        self.subjectName = subjectName
        self.subjectAge = subjectAge
        self.subjectGender = subjectGender
        self.subjectRace = subjectRace
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

    def save_session_data(self, rankings):
        ## TODO: save the data gathered in the session 
        ## to some format like csv or other
        cols = [0,1,2,3]
        featureNames = self.queries[0].featureNames
        cols.extend(featureNames)
        cols.append("Model Output")

        indx = ['SubjectInfo', "Model Rankings"]
        indx.extend(range(len(self.queries)))

        df = pd.DataFrame(index=indx, columns=cols)
        df.iloc[0,0] = self.subjectName
        df.iloc[0,1] = self.subjectAge
        df.iloc[0,2] = self.subjectGender
        df.iloc[0,3] = self.subjectRace

        for i,ranking in enumerate(rankings):
            df.iloc[1,i] = ranking

        for i, query in enumerate(self.queries):
            df.iloc[i+2,4:-1] = query.features
            df.iloc[i+2, -1] = query.output
            df.iloc[i+2, 3] = query.model

        df.to_csv(f"session_results/{self.subjectName}.csv")

        return

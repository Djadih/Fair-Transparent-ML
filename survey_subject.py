import pandas as pd

class RawQuery():
    def __init__(self, modelType=None, inputs=None, output=None):
        self.model = modelType
        self.inputs = inputs
        self.output = output
        return

    def set_model(self, model):
        self.model = model
    def set_inputs(self, inputs):
        self.inputs = inputs
    def set_output(self, output):
        self.output = output

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
        self.raw_queries = []
        return

    def log_completed_query(self, raw_query, query):
        self.raw_queries.append(raw_query)
        self.queries.append(query)
        return

    def print_all_queries(self):
        # mostly for debugging purposes
        print("Raw Queries")
        for query in self.raw_queries:
            print(f"Model: {query.model}")
            print(f"Raw Inputs: {query.inputs}")
            print(f"Model Output: {query.output}\n")

        print("Queries")
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

    def save_session_raw_queries(self):
        inputFeatureNames = list(self.raw_queries[0].inputs.keys())

        cols = inputFeatureNames
        cols.insert(0, "Model")
        cols.append("Predicted Income (Model Output)")

        raw_query_list = []
        for query in self.raw_queries:
            formattedQuery = dict()
            formattedQuery["Model"] = query.model
            formattedQuery.update(query.inputs)
            formattedQuery["Predicted Income (Model Output)"] = query.output
            raw_query_list.append(formattedQuery)

        df = pd.DataFrame(raw_query_list)
        df = df.sort_values(by=["Model"])

        df.to_csv(f"session_results/{self.subjectName}_log.csv")


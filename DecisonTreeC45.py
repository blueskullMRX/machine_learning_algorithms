import numpy as np
import pandas as pd


class DecisionTree :
    def __init__(self,min_data=10,max_depth = 5):
        self.tree = None
        self.min_data = min_data
        self.max_depth = max_depth

    #main methods :

    def fit(self,data,target_feature='class',current_depth = 0) : 
        #stopping conditions : data size , no more features to test on , max depth acheived
        if data.shape[0] < self.min_data or len(data[target_feature].unique()) == 1 or current_depth>=self.max_depth:
            return data[target_feature].mode()[0]
        
        #stopping condition : highest chi2 feature returned none
        best_feature = DecisionTree.find_highest_gain_ratio_value(data, target_feature)
        if best_feature is None:
            return data[target_feature].mode()[0]  
        
        best_feature_values= data[best_feature].unique()

        tree = {best_feature:{}}
        for best_feature_value in best_feature_values :
            tree[best_feature][best_feature_value] = self.fit(DecisionTree.divide_data(data,best_feature)[best_feature_value],target_feature,current_depth=current_depth+1)
            
        self.tree = tree
    
    def predict(self,data) :
        test_res_predicted = []
        #fun to classify single sample and append it to test_res_predicted
        def predict_row(row):
            test_res_predicted.append(DecisionTree.classify(self.tree,row))
        #apply predict_row on all data
        data.apply(predict_row,axis=1)
        return test_res_predicted
    
    @staticmethod
    def accuracy(predicted,result) :
        count = 0
        for i in range(len(predicted)) :
            if predicted[i] == result.values[i] :
                count += 1
        return count/len(predicted)
    
    @staticmethod
    def classify(tree, sample):
        #if sample is a leaf returns the value 1 or 0
        if not isinstance(tree, dict): 
            return tree

        feature = next(iter(tree))
        value = sample[feature]
        #if value didnt appear in training data
        if value not in tree[feature]:
            return None
        subtree = tree[feature][value]
        return DecisionTree.classify(subtree, sample)
    


    #calculators :
    
    @staticmethod
    def entropy(dataset,feature_name,target_name) :
        feature = dataset[feature_name].to_numpy()
        target = dataset[target_name].to_numpy()

        unique_features = np.unique(feature)
        unique_targets = np.unique(target)

        counts = np.zeros((len(unique_features), len(unique_targets)), dtype=int)

        for i, f_val in enumerate(unique_features):
            for j, t_val in enumerate(unique_targets):
                counts[i, j] = np.sum((feature == f_val) & (target == t_val))


        total_entropy = 0
        data_size = dataset.shape[0]
        for i in range(counts.shape[0]) :
            total_count = sum(counts[i])
            entropy_each_value = 0
            for j in range(counts.shape[1]) :
                p = counts[i][j] / total_count
                if p != 0 :
                    entropy_each_value += -p * np.log2(p)

            feature_value_count = dataset[dataset[feature_name]==unique_features[i]].shape[0]
            total_entropy += feature_value_count/data_size * entropy_each_value
            
        return total_entropy

    @staticmethod
    def entropy_main(dataset,target_name) :
        target = dataset[target_name].to_numpy()
        unique_targets,counts = np.unique(target,return_counts=True)
        
        total = sum(counts)
        entropy = 0
        for i in range(len(unique_targets)) :
            p = counts[i]/total
            entropy += -p*np.log2(p)

        return entropy

    @staticmethod
    def gain(dataset,feature,target) :
        return DecisionTree.entropy_main(dataset,target) - DecisionTree.entropy(dataset,feature,target)

    @staticmethod
    def si(dataset,feature_name):
        unique_features,counts = np.unique(dataset[feature_name],return_counts = True)
        total = sum(counts)
        si = 0
        for i in range(len(unique_features)) :
            p = counts[i]/total
            si += -p*np.log2(p)
        return si

    @staticmethod
    def gain_ratio(dataset,feature,target) :
        gn = DecisionTree.gain(dataset,feature,target)
        s = DecisionTree.si(dataset,feature)
        if s == 0 : return 0
        return gn/s
    
    @staticmethod
    def find_highest_gain_ratio_value(data, target_column):
        features = data.columns.tolist()
        features.remove(target_column)

        max_gain = -1
        max_gain_feature = None

        for feature in features :
            gain = DecisionTree.gain_ratio(data,feature,target_column)
            if gain > max_gain :
                max_gain = gain
                max_gain_feature = feature
        return max_gain_feature

    @staticmethod
    def divide_data(data,feature):
        feature_values = data[feature].unique()
        datas = {}
        for feature_value in feature_values :
            data_byvalue = data[data[feature] == feature_value]
            data_byvalue = data_byvalue.drop(feature,axis='columns')
            datas[feature_value] = data_byvalue
        return datas
    


    #categorizing data :

    @staticmethod
    def detect_continuous_columns(df, threshold=10):
        continuous_columns = []
        for column in df.columns:

            if np.issubdtype(df[column].dtype, np.number):
                unique_values = df[column].nunique()
                total_values = len(df[column])

                if unique_values > threshold or unique_values / total_values > 0.5:
                    continuous_columns.append(column)
        return continuous_columns

    @staticmethod
    def best_split_value(dataset,feature_name,target):
        feature = dataset[feature_name].to_numpy()
        unique_features = np.unique(feature)

        gain_ratio_max = -1
        gain_ratio_max_value = None

        for i in range(len(unique_features)) :
            categorize = np.zeros(feature.shape)

            for j,value in enumerate(feature) :
                categorize[j] = 1 if value > unique_features[i] else 0

            tg = dataset[target].to_numpy()
            test = np.column_stack((categorize, tg)) 
            test_data = pd.DataFrame(data=test,columns=['feature','target'])

            gain_rat = DecisionTree.gain_ratio(test_data,'feature','target')
            if gain_rat > gain_ratio_max : 
                gain_ratio_max = gain_rat
                gain_ratio_max_value = unique_features[i]
                
        return gain_ratio_max_value
    
    @staticmethod
    def to_category(data,feature,target):
        best = DecisionTree.best_split_value(data,feature,target)
        return data[feature].apply(lambda x: 1 if x>best else 0)



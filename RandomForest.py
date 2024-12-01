import numpy as np
import pandas as pd

#from DecisonTreeC45 import entropy,entropy_main,gain,si,gain_ratio,best_split_value,to_category,detect_continuous_columns,find_highest_gain_ratio_value,divide_data,classify,predict,accuracy
from DecisonTreeC45 import DecisionTree

class RandomForest :
    def __init__(self,nbr_trees=5,tree_min_data=10,tree_max_depth = 5):
        self.tree_min_data = tree_min_data
        self.tree_max_depth = tree_max_depth
        self.nbr_trees = nbr_trees
        self.trees = []
        self.oob = None


    #main methods :

    def train_random_forest(self,data,target_feature='class'):
        trees = []
        oob = []
        for i in range(self.nbr_trees):
            sub_data,test_sub_data = self.bootstrap_sample(data)
            oob.append(test_sub_data)
            trees.append(self.build_tree(sub_data,target_feature))
        test_data = pd.concat(oob, axis=0, ignore_index=True)
        test_data.drop_duplicates(inplace=True)
        self.trees = trees,
        self.oob = test_data
    
    def build_tree(self,data,target_feature='class',current_depth = 1) : 
        #stopping conditions : data size , no more features to test on , max depth acheived
        if data.shape[0] < self.tree_min_data or len(data[target_feature].unique()) == 1 or current_depth==self.tree_max_depth:
            return data[target_feature].mode()[0]
        
        #random forest edit !!!
        features = data.columns.to_list()
        features.remove(target_feature)
        selected_features = RandomForest.random_feature_selection(features)
        selected_features.append(target_feature)
        selected_data = data[selected_features]
        selected_data

        #stopping condition : highest chi2 feature returned none
        best_feature = DecisionTree.find_highest_gain_ratio_value(selected_data, target_feature)
        if best_feature is None:
            return data[target_feature].mode()[0]  
        
        best_feature_values= data[best_feature].unique()

        tree = {best_feature:{}}
        for best_feature_value in best_feature_values :
            tree[best_feature][best_feature_value] = self.build_tree(DecisionTree.divide_data(data,best_feature)[best_feature_value],target_feature,current_depth=current_depth+1)
            
        return tree

    def predict_random_forest(self,data):
        results = []
        for tree in self.trees : 
            results.append(DecisionTree.predict(tree,data))
        rs = np.column_stack(results)
        rs_data = pd.DataFrame(data=rs,columns=range(len(self.trees)))
        most_common_values = rs_data.mode(axis=1)[0].values
        #most_common_values = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=rs)
        #most_common_values = most_frequent_in_rows(rs)
        return most_common_values
    

    #CALCULATORS : 

    @staticmethod
    def random_feature_selection(features):
        num_features_to_select = int(np.ceil(np.sqrt(len(features))))
        selected_features = np.random.choice(features, size=num_features_to_select, replace=False)
        return selected_features.tolist()

    @staticmethod
    def bootstrap_sample(data): 
        n = len(data)
        
        bootstrap_indices = np.random.choice(range(n), size=n, replace=True)
        
        oob_indices = list(set(range(n)) - set(bootstrap_indices))
        
        bootstrap = data.iloc[bootstrap_indices]
        out_of_bag = data.iloc[oob_indices]
        
        return bootstrap, out_of_bag


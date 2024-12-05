import numpy as np
import pandas as pd

from DecisionTreeC45 import entropy,entropy_main,gain,si,gain_ratio,best_split_value,to_category,detect_continuous_columns,find_highest_gain_ratio_value,divide_data,classify,predict,accuracy

def random_feature_selection(features):
    num_features_to_select = int(np.ceil(np.sqrt(len(features))))
    selected_features = np.random.choice(features, size=num_features_to_select, replace=False)
    return selected_features.tolist()


def build_tree(data,min_data=10,target_feature='class',max_depth = 5,current_depth = 1) : 
    #stopping conditions : data size , no more features to test on , max depth acheived
    if data.shape[0] < min_data or len(data[target_feature].unique()) == 1 or current_depth==max_depth:
        return data[target_feature].mode()[0]
    

    #random forest edit !!!
    features = data.columns.to_list()
    features.remove(target_feature)
    selected_features = random_feature_selection(features)
    selected_features.append(target_feature)
    selected_data = data[selected_features]
    selected_data

    #stopping condition : highest chi2 feature returned none
    best_feature = find_highest_gain_ratio_value(selected_data, target_feature)
    if best_feature is None:
        return data[target_feature].mode()[0]  
    
    best_feature_values= data[best_feature].unique()

    tree = {best_feature:{}}
    for best_feature_value in best_feature_values :
        tree[best_feature][best_feature_value] = build_tree(divide_data(data,best_feature)[best_feature_value],min_data,target_feature,current_depth=current_depth+1,max_depth=max_depth)
        
    return tree


def bootstrap_sample(data): 
    n = len(data)
    
    bootstrap_indices = np.random.choice(range(n), size=n, replace=True)
    
    oob_indices = list(set(range(n)) - set(bootstrap_indices))
    
    bootstrap = data.iloc[bootstrap_indices]
    out_of_bag = data.iloc[oob_indices]
    
    return bootstrap, out_of_bag


def train_random_forest(data,nbr_trees=5,tree_min_data=10,target_feature='class',tree_max_depth = 4):
    trees = []
    oob = []
    for i in range(nbr_trees):
        sub_data,test_sub_data = bootstrap_sample(data)
        oob.append(test_sub_data)
        trees.append(build_tree(sub_data,tree_min_data,target_feature,tree_max_depth))
    test_data = pd.concat(oob, axis=0, ignore_index=True)
    test_data.drop_duplicates(inplace=True)
    return trees,test_data


def predict_random_forest(trees,data):
    results = []
    for tree in trees : 
        results.append(predict(tree,data))
    rs = np.column_stack(results)
    rs_data = pd.DataFrame(data=rs,columns=range(len(trees)))
    most_common_values = rs_data.mode(axis=1)[0].values
    #most_common_values = np.apply_along_axis(lambda row: np.bincount(row).argmax(), axis=1, arr=rs)
    #most_common_values = most_frequent_in_rows(rs)
    return most_common_values
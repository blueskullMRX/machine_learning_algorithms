import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



#based on Filter
def corr_filter(df,target,min_cor=0.35):
    corr = df.corr()
    target_corr = corr[target].abs()
    #target_corr = target_corr.drop(target)
    selected_features = target_corr[target_corr >= min_cor].index.tolist()
    selected_features.remove(target)
    return selected_features

#based on filter : advanced uses corr bewteen features too
def corr_filter_between(df,target,min_cor=0.35):
    corr = df.corr()
    target_corr = corr[target].abs()
    #target_corr = target_corr.drop(target)
    selected_features = target_corr[target_corr >= min_cor].index.tolist()

    features = selected_features
    features.remove(target)
    corr_between = df[features].corr()
    for feature in selected_features : 
        if feature not in features : continue
        target_corr_between = corr_between[feature].abs()
        delete_features = target_corr_between[target_corr_between >= 1-min_cor].index.tolist()
        for x in delete_features :
            if x in features : features.remove(x)
    return features

#based on Wrapper
from RandomForest import train_random_forest,predict_random_forest,accuracy,gain_ratio

def to_accuracy_rf(data,target):# rf= random forest

    train,test = train_test_split(data, test_size=0.2, random_state=42)
    
    train = pd.DataFrame(data=train,columns=data.columns)
    test = pd.DataFrame(data=test,columns=data.columns)
    
    trees,_ = train_random_forest(train,tree_max_depth=4,nbr_trees=8)
    pred = predict_random_forest(trees,test)
    return accuracy(pred,test[target])

from LogisticRegression import train_lr,predict
from sklearn.preprocessing import StandardScaler

def to_accuracy_lr(data,target):# lr= logistic regression
    x = data.drop(target, axis='columns').values
    y = data[target].values
    y_df = data[target]
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)
    y_test = pd.DataFrame(data=y_test,columns=[target])

    w,b = train_lr(x_train,y_train,epsilon=0.00001,max_iteration = 3000)
    pred = predict(x_test,w,b)
    return accuracy(pred,y_test)


def backward_selection_rf(data,target,max_columns=50):# rf= random forest
    features = data.columns.to_list()
    remaining_features = features
    acc = to_accuracy_rf(data,target)
    print(acc)
    print(len(remaining_features))
    while len(remaining_features) > max_columns :
        print(len(remaining_features))
        print(acc)
        for feature in features :
            if feature == target :
                continue
            temp = remaining_features
            temp.remove(feature)
            temp_acc = to_accuracy_rf(data[temp],target)
            if temp_acc >= acc :
                acc = temp_acc
                remaining_features = temp
    remaining_features.remove(target)
    return remaining_features

def backward_selection_lr(data,target,max_columns=50):# lr= logistic regression
    features = data.columns.to_list()
    remaining_features = features
    acc = to_accuracy_lr(data,target)
    print(acc)
    print(len(remaining_features))
    while len(remaining_features) > max_columns :
        print(len(remaining_features))
        print(acc)
        for feature in features :
            if feature == target :
                continue
            temp = remaining_features
            temp.remove(feature)
            temp_acc = to_accuracy_lr(data[temp],target)
            if temp_acc >= acc :
                acc = temp_acc
                remaining_features = temp
    remaining_features.remove(target)
    return remaining_features


#based on embedded
def recursive_feature_elimination_rf(train_data,target,min_feature=1): # rf= random forest
    data = train_data.copy()
    features = data.columns.to_list()
    features.remove(target)

    importances = {}
    for feature in features :
        importances[feature] = gain_ratio(data,feature,target)

    while len(features) > min_feature :
        data = data[features]
        data[target] = train_data[target]
        
        least_important_feature = min(importances, key=importances.get)        
        
        features.remove(least_important_feature)
        del importances[least_important_feature]

    return features


def recursive_feature_elimination_lr(train_data,target,min_feature=1): # lr= logistic regression
    data = train_data.copy()
    features = data.columns.to_list()
    features.remove(target)

    train_x = train_data[features].values
    train_y = train_data[target].values
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)

    importances = {}
    w,_= train_lr(train_x,train_y,epsilon=0.00001,max_iteration = 3000)
    for i,feature in enumerate(features) :
        importances[feature] = np.abs(w[i])

    while len(features) > min_feature :
        data = data[features]
        data[target] = train_data[target]
        
        least_important_feature = min(importances, key=importances.get)        
        
        features.remove(least_important_feature)
        del importances[least_important_feature]

    return features
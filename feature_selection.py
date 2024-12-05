import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


#to accuracy functions :
from RandomForest import accuracy
from LogisticRegression import train_lr,predict
from sklearn.preprocessing import StandardScaler
def to_accuracy_lr(data,target,learning_rate=0.01,epsilon=0.00001,max_iteration = 2000,test_size=0.2):# lr= logistic regression
    x = data.drop(target, axis='columns').values
    y = data[target].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=test_size, random_state=42)
    y_test = pd.DataFrame(data=y_test,columns=[target])

    w,b = train_lr(x_train,y_train,learning_rate,epsilon,max_iteration)
    pred = predict(x_test,w,b)
    return accuracy(pred,y_test)

#based on Filter
def corr_filter(df,target,min_cor=0.35):
    corr = df.corr()
    target_corr = corr[target].abs()
    #target_corr = target_corr.drop(target)
    selected_features = target_corr[target_corr >= min_cor].index.tolist()
    selected_features.remove(target)
    return selected_features

#based on Wrapper
def backward_selection_lr(data,target):# lr= logistic regression
    features = data.columns.to_list()
    remaining_features = list(features)
    acc = to_accuracy_lr(data,target)
    #print(f"Number of feature remaining : {len(remaining_features)}")
    while True :
        features = list(remaining_features)
        feature_count = len(remaining_features)
        i = 0
        for feature in features :
            i+=1
            #print(i,end=' ')
            if feature == target :
                continue
            temp = list(remaining_features)
            temp.remove(feature)
            temp_acc = to_accuracy_lr(data[temp],target)
            if temp_acc >= acc :
                acc = temp_acc
                remaining_features = list(temp)
        #print(f"Number of feature remaining : {len(remaining_features)}")
        if feature_count == len(remaining_features) : break
    remaining_features.remove(target)
    return remaining_features


#based on embedded
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
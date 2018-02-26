import numpy as np
import pandas as pd

#Split a dataframe based on a feature and a feature value
def df_split(df, feature, value):
    left = df[df[feature]<=value]
    right = df[df[feature]>value]
    return left, right #(left df, right df)

#entropy of a dictionary
def entropy(a_dict):
    s = 0    
    for i in a_dict.keys():
        pi = a_dict[i]/sum(a_dict.values())
        s -= pi * np.log2(pi)
    return s

#information gain of a dictionary wrt a target feature
def infogain(df, parts, target): 
    dict0 = dict(df[target].value_counts())
    dict1 = dict(parts[0][target].value_counts())
    dict2 = dict(parts[1][target].value_counts())
    s0 = entropy(dict0)
    s1 = entropy(dict1)
    s2 = entropy(dict2)
    return s0-s1*sum(dict1.values())/sum(dict0.values())-s2*sum(dict2.values())/sum(dict0.values())

# Best value to split a feature wrt a target feature
def best_value_to_split(df, feature, target):    
    d = {}
    count = 0
    summ = 0
    for i in range(min(set(df[feature].values)),max(set(df[feature].values))):        
        d[i] = infogain(df, df_split(df, feature, i), target)
    for key in d.keys():
        if d[key] == max(d.values()):
            count+=1
            summ += key
    return summ/count

# returns a feature which is best to split on a df and a target
def best_feature_to_split(df, target):
    l = {}
    for i in df.columns:
        if i != target:
            l[i] = infogain(df, df_split(df, i, best_value_to_split(df, i, target)), target) 
    for key in l.keys():
        if l[key] == max(l.values()):
            return key

# prints out the leafs of the tree
def build_tree(df, target):
    feature = best_feature_to_split(df, target)
    v1 = best_value_to_split(df, feature, target)
    part1, part2 = df_split(df, feature, v1)
    ent = entropy(dict(df[target].value_counts()))
    samples = len(df.index)
    value = dict(df[target].value_counts())
    print(feature, v1, ent, samples, len(part1.index), len(part2.index), value)
    if entropy(dict(part1[target].value_counts())) == 0.0:
        ent1 = entropy(dict(part1[target].value_counts()))
        samples1 = len(part1.index)
        value1 = dict(part1[target].value_counts())
        print (feature, v1, ent1, samples1, value1)
    else:
        return build_tree(part1, target)
    if entropy(dict(part2[target].value_counts())) == 0.0:
        ent2 = entropy(dict(part2[target].value_counts()))
        samples2 = len(part2.index)
        value2 = dict(part2[target].value_counts())
        print (feature, v1, ent2, samples2, value2)   
    else:
        return build_tree(part2, target)
 
# a toy example
df = pd.DataFrame({'Age':  [17,64,18,20,38,49,55,25,29,31,33], 
                      'Salary': [25,80,22,36,37,59,74,70,33,102,88], 
             'Loan Default': [1,0,1,0,1,0,0,1,1,0,1]})

build_tree(df, "Loan Default")

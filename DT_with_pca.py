from random import seed
from random import randrange
from csv import reader
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import math
import sys
import numpy as np

impurity_compute_algo = 0

# Calculate the entropy for a split dataset
def entropy(groups, classes):
    score = 0.0
    p = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        # score the group based on the score for each class
        for class_val in classes:
            p+= [row[-1] for row in group].count(class_val)
            ratio = p /size
            if ratio == 0.0:
                continue
            temp= (-p/size) * math.log(p/size,2)
            score += temp
            p = 0.0
    return score


# Calculate the misclassification error for a split dataset
def misclassification_error(groups, classes):
    score = 0.0
    p = 0.0
    x=[]
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        # score the group based on the score for each class
        for class_val in classes:
            p+= [row[-1] for row in group].count(class_val)
            ratio = p /size
            x.append(ratio)
            p = 0.0
        maximum = max(x)
        score += (1 - maximum)
    return score

# Calculate the Gini index for a split dataset
def gini_index_impurity(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def confusion_matrixt(y_true, y_pred):
    print "confusion matrix:"
    print(confusion_matrix(y_true, y_pred))

 
# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
 
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0
 
# Evaluate an algorithm using a cross validation split
def test_decision_tree_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        #confusion_matrixt(actual, predicted)
        scores.append(accuracy)
    return scores
 
# Split a dataset based on an attribute and an attribute value
def split_node(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right
 
 
# Select the best split point for a dataset
def call_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = split_node(index, row[index], dataset)
            #Select impurity compute algo
            if impurity_compute_algo == 1:
                impurity = entropy(groups, class_values)
            elif impurity_compute_algo == 2:
                impurity = misclassification_error(groups, class_values)
            else:
                impurity = gini_index_impurity(groups, class_values)
            if impurity < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], impurity, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}
 
# Create a terminal node value
def leaf_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    #del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = leaf_node(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = leaf_node(left), leaf_node(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = leaf_node(left)
    else:
        node['left'] = call_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = leaf_node(right)
    else:
        node['right'] = call_split(right)
        split(node['right'], max_depth, min_size, depth+1)
 
# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = call_split(train)
    split(root, max_depth, min_size, 1)
    return root
 
# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
 
# Build decision tree
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)
 
def load_csv(filename):
    file = open(filename, "rb")
    lines = reader(file)
    dataset = list(lines)
    return dataset
 
# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

##########################
# Default parameters unless user do input via command line
##########################

seed(10)
n_folds = 5
max_depth = 8
min_size = 10
nf       = 5
##########################
# 1 = Select Entropy
# 2 = Select misclassification error
# 3 = Select gini index impurity
##########################
impurity_compute_algo = 3



# load and prepare data
filename = 'clean.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)


label=[]
dataset_tmp=[]
temp2=[]
for row in dataset:
    label.append(row[-1])
    dataset_tmp.append(row[:-1])

#Apply dimensionality reduction
pca = PCA(n_components=nf)
pca.fit(dataset_tmp)
dataset_new=pca.transform(dataset_tmp)
temp1 = dataset_new.tolist()

for i, row in zip(xrange(len(label)), temp1):
    temp2.append(np.append(row,label[i]))

temp2=(np.array(temp2)).tolist()

if len(sys.argv)==4:
    n_folds    = int(sys.argv[1])
    max_depth  = int(sys.argv[2])
    min_size   = int(sys.argv[3])
count=3
while(count < 33):
    scores = test_decision_tree_algorithm(temp2, decision_tree, n_folds, count, min_size)
    print('Scores: %s' % scores)
    print('%d Mean Accuracy: %.3f' % (count, sum(scores)/float(len(scores))))
    count+=2

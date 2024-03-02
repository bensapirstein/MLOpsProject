import pandas as pd
import numpy as np
import os
pd.set_option('display.max_rows', None)

# DT
from sklearn.metrics import accuracy_score
import dtreeviz
from sklearn.tree import DecisionTreeClassifier, export_text
import itertools

###################### Decision Tree #############
# Src: https://towardsdatascience.com/train-a-regression-model-using-a-decision-tree-70012c22bcc1
# Src: https://towardsdatascience.com/an-exhaustive-guide-to-classification-using-decision-trees-8d472e77223f
def fit_DT(df, predictors):
    """ Fit a classification decision tree and return key elements """

    X = df[predictors] 
    y = df['accuracy_bool'] 

    model = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=1)
    model.fit(X, y)

    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    return model, preds, acc, X

def get_DT_rules(model, predictors):
    """ Return the rules of a decision tree """

    r = export_text(model, feature_names=predictors)
    return r

def visualize_DT(df, model, predictors, fname='tree', outdir='trees'):
    """ Visualize tree with data plots """
    X = df[predictors]
    acc = df['accuracy_bool'].values
    viz = dtreeviz.model(model, X, acc,
                    target_name="accuracy_bool",
                    feature_names=predictors,
                    class_names=['True','False'] if acc[0] else ['False','True'])

    v = viz.view()
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    v.save(f'{outdir}/{fname}.svg')

def return_dt_split(model, col, accuracy, col_2=None, impurity_cutoff=1.0, n_datapoints_cutoff=5, acc_cutoff=0.03):
    """
    Return all indices of col that meet the following criteria:
    1. Leaf has accuracy lower that baseline by acc_cutoff
    2. Split size > n_datapoints_cutoff 

    :param model: SKLearn classification decision tree model
    :param col: (pd.Series) column used to split on
    :param accuracy: (pd.Series) column corresponding to correct/incorrect classification
    :param col_2: (pd.Series) column to be used for interactions
    :param impurity_cutoff: (float) requirement for entropy/gini of leaf
    :param n_datapoints_cutoff: (int) minimum n in a final node to be returned
    :param acc_cutoff: (float) accuracy cutoff for returning float
    :return: (dict[node_idx, indices]) where indices corresponds to the col that meet the above criteria
    """

    # get leaf ids and setup
    df = pd.concat([col, col_2], axis=1) if col_2 is not None else pd.DataFrame(col)
    leaf_id = model.apply(df)

    t = model.tree_
    baseline_acc = np.mean(accuracy)
    
    # get indices of leaf ids that meet criteria
    keeps_1 = {i for i,v in enumerate(t.n_node_samples) if v > n_datapoints_cutoff} # sample size
    keeps_2 = {i for i,v in enumerate(t.impurity) if v <= impurity_cutoff} # sample size
    keeps = keeps_1 & keeps_2

    # store all data and corresponding index
    node_indices = {}
    slice_acc = -1
    for idx in keeps:
        node_indices[idx] = [i for i,v in enumerate(leaf_id) if v == idx]

        # remove non-low-accuracy areas and empty lists
        slice_acc = [x[1] / sum(x) for x in t.value[idx]] 
        if baseline_acc - slice_acc < acc_cutoff or node_indices[idx] == []:
            del node_indices[idx]
            slice_acc = None

    return (f'{col.name}{"-"+col_2.name if col_2 is not None else ""}', list(node_indices.keys()), list(node_indices.values()), slice_acc)

###################
# Run
###################

###################### Run Helpers #############
def run_data_search(df, name=None):
    """ 
    Iterate over data columns and perform decision tree analysis
    Save DT viz and print problematic indices

    :param df: (pd.DataFrame) of raw data with correct/incorrect classification column
    """

    acc_col = df['accuracy_bool']

    # store for output
    bivariate_acc = []

    pairs = list(itertools.combinations(set(df) - set(['accuracy_bool']), 2))

    for col1, col2 in pairs:
        print(f'Running {col1} and {col2}')
        c1, c2 = df[col1], df[col2]
        predictors = [col1,col2]
        model, *_, X = fit_DT(df, predictors=predictors)

        outdir = 'trees' if name is None else f'trees/{name}'
        visualize_DT(df, model, predictors=predictors, fname=f'{col1}-{col2}', outdir=outdir)

        bivariate_acc.append(return_dt_split(model, c1, acc_col, c2))

    return clean_output(bivariate_acc)

def clean_output(a):
    """ 
    Take list of outputs of DT interactions and return sorted 
    value by accuracy drop.
    """

    names, indices, accuracies, method = [], [], [], []

    for x in a:
        names.append(x[0])
        indices.append(x[2])
        accuracies.append(x[3])
        method.append('DT')

    out = pd.DataFrame(dict(names=names, indicies=indices, accuracies=accuracies, method=method))
    # out['original_index'] = out.index
    out.sort_values(by=['accuracies'], inplace=True)
    out.index = range(len(out.index))
    out.dropna(inplace=True)

    return out

def feature_engineering(out2, train_x, valid_x, test_x):
    new_train_x = train_x.copy()
    new_val_x = valid_x.copy()
    new_test_x = test_x.copy()
    counter = 0

    cols = []
    # Take slices according to explainability results
    for x,y in zip(out2["names"], out2["indicies"]):
        for i in range(len(y)):
            col1 = x.split("-")[0]
            col2 = x.split("-")[1]
            z = train_x[col1].iloc[y[i]]
            min_value1 = z.min()
            max_value1 = z.max()

            z = train_x[col2].iloc[y[i]]
            min_value2 = z.min()
            max_value2 = z.max()

            new_col = "x" + str(counter)
            cols.append(new_col)
            counter += 1

            new_train_x[new_col] = 0
            x1 = new_train_x[col1] >= min_value1
            x2 = new_train_x[col1] <= max_value1

            x3 = new_train_x[col2] >= min_value2
            x4 = new_train_x[col2] <= max_value2
            new_train_x[new_col].iloc[new_train_x[x1][x2][x3][x4].index.to_numpy()] = 1


            new_val_x[new_col] = 0
            x1 = new_val_x[col1] >= min_value1
            x2 = new_val_x[col1] <= max_value1
            x3 = new_val_x[col2] >= min_value2
            x4 = new_val_x[col2] <= max_value2

            new_val_x[new_col].iloc[new_val_x[x1][x2][x3][x4].index.to_numpy()] = 1

            new_test_x[new_col] = 0
            x1 = new_test_x[col1] >= min_value1
            x2 = new_test_x[col1] <= max_value1
            x3 = new_test_x[col2] >= min_value2
            x4 = new_test_x[col2] <= max_value2
            new_test_x[new_col].iloc[new_test_x[x1][x2][x3][x4].index.to_numpy()] = 1

    return new_train_x, new_val_x, new_test_x, cols
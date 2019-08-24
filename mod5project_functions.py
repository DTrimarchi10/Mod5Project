import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, accuracy_score

#Function used to plot histograms
def plot_histograms(df,
                    column_names,
                    nrows,
                    ncols,
                    label_name='target',
                    bins=30,
                    figsize=(15,10)):
    """
    This function will provide a plot (with subplots) for the dataframe and desired 
    features. It will plot the binary target values separately on the feature plot.
    
    INPUTS:
    df           = The dataframe containing the data to be plotted.
    column_names = The feature in df that will be plotted.
    nrows, ncols = The number of rows and columns in the subplot grid.
    label_name   = The name of the target variable column.
    bins         = The number of bins to use in the histogram.
    figsize      = The total size of the figure.
    
    """
    fig = plt.figure(figsize=figsize)
    subplot_index = 1
    for col in column_names:
        ax = fig.add_subplot(nrows, ncols, subplot_index)
        target_true = df[df[label_name]==1][col]
        target_false = df[df[label_name]==0][col]
        if target_true.count():
            ax.hist(target_true, bins=bins,alpha=0.5)
        if target_false.count():
            ax.hist(target_false, bins=bins,alpha=0.5)
        ax.set_title(col)
        subplot_index+=1
    plt.tight_layout()
    
    return

def get_feature_importances(estimator, df, num_features=10):
    """
    This function will print the feature importances from the provided model
    and dataframe.
    
    INPUTS:
    estimator    = The model that has been fit.
    df           = The dataframe containing model features.
    num_features = The number of features to print.
    """
    importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1]
    if len(importances) < num_features:
        num_features = len(importances)
    print("Feature ranking:")
    for idx, f_idx in enumerate(indices[0:num_features]):
        print(f"{idx+1}: Score {round(importances[f_idx],3)}, Feature Name: {df.columns[f_idx]}")
    return

def print_model_stats(estimator, X_train, X_test, y_train, y_test):
    """
    This function prints the accuracy and precision score for the provided
    model using the training and test datasets provided.
    """
    y_pred_train = estimator.predict(X_train)
    y_pred_test  = estimator.predict(X_test)
    print(" TRAINING:  Accuracy:",round(accuracy_score(y_train,y_pred_train),4),
          " Precision:",round(precision_score(y_train,y_pred_train),4))
    print(" TEST:      Accuracy:",round(accuracy_score(y_test,y_pred_test),4),
          " Precision:",round(precision_score(y_test,y_pred_test),4))   
    return
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_pruning_graph(X_train, y_train, clf):
    # clf is the model you are using
    # Would be declared in the main function like :
    # clf = DecisionTreeClassifier(random_state=0, criterion="entropy", ccp_alpha=0.01,
    #                              class_weight='balanced',
    #                              splitter = "random"
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post", color = 'black', ls = '--')
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    #Gets stored
    fig.savefig("ImpurityVSalpha.png")
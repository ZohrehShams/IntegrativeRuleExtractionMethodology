import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.utils import class_weight
from rules.tree import decisionTree_Ruleset, randomForest_Ruleset

# run decision tree
def run_dt(X_train, y_train, X_test, y_test, feature_names_to_id_map, output_classes, max_depth):
    cw = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

    if max_depth is None:
        decision_tree = DecisionTreeClassifier(class_weight=cw, criterion='entropy', random_state=42)
    else:
        decision_tree = DecisionTreeClassifier(class_weight=cw, max_depth=max_depth, criterion ='entropy', random_state=42)

    decision_tree = decision_tree.fit(X_train, y_train)
    predicted = decision_tree.predict(X_test)
    dt_accuracy = accuracy_score(y_test, predicted)

    dt_AUC = 0
    number_of_labels = len(np.unique(y_train))
    if number_of_labels == 2:
        fpr, tpr, thresholds = roc_curve(y_test, predicted)
        dt_AUC = auc(fpr, tpr)
    else:
        for i in range(number_of_labels):
            fpr, tpr, thresholds = roc_curve(y_test == i, predicted == i)
            dt_AUC  += auc(fpr, tpr)
        dt_AUC  /= number_of_labels

    rules = decisionTree_Ruleset(decision_tree, feature_names_to_id_map, output_classes)

    return dt_accuracy, dt_AUC, rules


# run random forest
def run_rf(X_train, y_train, X_test, y_test, feature_names_to_id_map, output_classes, n_estimators, max_depth):
    cw = dict(enumerate(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)))

    random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=cw, random_state=42)
    random_forest = random_forest.fit(X_train, y_train)
    predicted = random_forest.predict(X_test)
    rf_accuracy = accuracy_score(y_test, predicted)
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    rf_AUC = auc(fpr, tpr)

    rules = randomForest_Ruleset(random_forest, feature_names_to_id_map, output_classes)

    return rf_accuracy, rf_AUC, rules
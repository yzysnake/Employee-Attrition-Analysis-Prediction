import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, f1_score

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from scipy import interp


def one_hot_encode(df):
    # Specify the columns to be one-hot encoded
    columns_to_encode = ['Department', 'MaritalStatus', 'Gender', 'JobRole',
                         'EducationField', 'Attrition', 'OverTime', 'BusinessTravel']

    # Initialize the OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first', dtype=np.integer)

    # Fit and transform the specified columns
    encoded_data = encoder.fit_transform(df[columns_to_encode])

    # Get new column names for the one-hot encoded variables
    encoded_columns = encoder.get_feature_names_out(columns_to_encode)

    # Create a DataFrame with the encoded data
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop the original columns from the DataFrame
    df_dropped = df.drop(columns=columns_to_encode)

    # Concatenate the original DataFrame (minus the to-be-encoded columns) with the new one-hot encoded DataFrame
    df_encoded = pd.concat([df_dropped, encoded_df], axis=1)

    return df_encoded


def min_max_scale(df):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    return df_scaled


def apply_pca(df, n_components=2):
    # Separating the target variable and features
    X = df.drop('Attrition_Yes', axis=1)
    y = df['Attrition_Yes']

    # Applying PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Creating a DataFrame for the PCA results
    pca_columns = [f'PCA_Component_{i}' for i in range(1, n_components + 1)]
    df_pca = pd.DataFrame(X_pca, columns=pca_columns)

    # Adding the target variable back to the PCA DataFrame
    df_pca['Attrition_Yes'] = y.reset_index(drop=True)

    return df_pca


def plot_learning_curves(model, X, y, cv):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)

    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 10),
                                                            scoring='f1')

    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Plotting the learning curves
    plt.plot(train_sizes, train_mean, '--', color="#111111", label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Drawing bands for the standard deviation
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Creating plot
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size"), plt.ylabel("F-1 Score"), plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_foldwise_scores(model, X, y, cv):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)

    # Ensure X and y are numpy arrays to simplify indexing in this context
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    train_scores = []
    val_scores = []

    # Perform cross-validation
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model.fit(X_train_fold, y_train_fold)

        y_train_pred = model.predict(X_train_fold)
        y_val_pred = model.predict(X_val_fold)

        train_f1 = f1_score(y_train_fold, y_train_pred)
        val_f1 = f1_score(y_val_fold, y_val_pred)

        train_scores.append(train_f1)
        val_scores.append(val_f1)

    # Plotting the scores across folds
    folds = range(1, cv.get_n_splits() + 1)
    plt.plot(folds, train_scores, 'o-', color="blue", label="Training F1 Score")
    plt.plot(folds, val_scores, 'o-', color="red", label="Validation F1 Score")

    # Adding plot details
    plt.title("F1 Scores across CV Folds")
    plt.xlabel("Fold"), plt.ylabel("F-1 Score")
    plt.xticks(list(folds), labels=list(folds))
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    new_cm = [[cm[1, 1], cm[0, 1]], [cm[1, 0], cm[0, 0]]]

    # Plot using seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(new_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Yes', 'No'], yticklabels=['Yes', 'No'])
    plt.title('Confusion Matrix (Yes Represents Attrition)')
    plt.ylabel('Predicted label')
    plt.xlabel('True label')
    plt.show()


def plot_precision_recall_curve(model, X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # If the model has a method to predict probabilities, use it; otherwise, use decision function
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)
        # ensure all scores are positive as precision_recall_curve expects positive probabilities
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())

    # Compute precision-recall pairs for different probability thresholds
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    # Plot the Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (area = {pr_auc:.2f})', lw=2, color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_random_forest_feature_importance(rf_model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66, stratify=y)

    # Extract feature importances
    feature_importances = rf_model.feature_importances_

    # Create a DataFrame for easier handling
    features = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': feature_importances
    })

    # Sort the features based on importance
    features_sorted = features.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(13, 10.4))
    bars = plt.barh(features_sorted['Feature'], features_sorted['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top

    # Adding the importance percentages at the end of each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.0005
        plt.text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width * 100:.2f}%', va='center')

    plt.show()

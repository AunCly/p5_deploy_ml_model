from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import cross_validate, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def split_train_data(data, target) :
    labels = data[target]
    train_data = data.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(train_data, labels, test_size=0.2, random_state=42, stratify=labels)

    data_train = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
    }

    return data_train

def benchmark(pipeline, train_data):
    scoring = {'recall': 'recall', 'f1': 'f1', 'roc_auc': 'roc_auc'}

    cv_results = cross_validate(
        pipeline,
        train_data['X_train'],
        train_data['y_train'],
        cv=5,
        scoring=scoring,
        return_train_score=True
    )

    print('--- Validation Fold Results ---')
    print(f"Validation Recall : {cv_results['test_recall']}, Recall moyen : {cv_results['test_recall'].mean()}")
    print(f"Validation F1-Scores {cv_results['test_f1']}, F1 moyen : {cv_results['test_f1'].mean()}")
    print(f"Validation ROC AUC : {cv_results['test_roc_auc']}, ROC moyen : {cv_results['test_roc_auc'].mean()}")

    print('\n--- Train Fold Results (Overfit Check) ---')
    print(f"Train Recall : {cv_results['train_recall']}, Recall moyen : {cv_results['train_recall'].mean()}")
    print(f"Train F1-Scores {cv_results['train_f1']}, F1 moyen : {cv_results['train_f1'].mean()}")
    print(f"Train ROC AUC : {cv_results['train_roc_auc']}, ROC moyen : {cv_results['train_roc_auc'].mean()}")

def train(pipeline, train_data):

    pipeline.fit(train_data['X_train'], train_data['y_train'])

    y_pred_train = pipeline.predict(train_data['X_train'])
    y_pred_test = pipeline.predict(train_data['X_test'])

    recall_train = recall_score(train_data['y_train'], y_pred_train)
    recall_test = recall_score(train_data['y_test'], y_pred_test)
    f1_train = f1_score(train_data['y_train'], y_pred_train)
    f1_test = f1_score(train_data['y_test'], y_pred_test)

    print(f'Training Résults : ')
    print(f"Train F1-Score: {f1_train}")
    print(f"Test F1-Score: {f1_test}")

    print(f'Testing Résults : ')
    print(f"Train Recall : {recall_train}")
    print(f"Test Recall : {recall_test}")

# Affiche les features d'importance
def show_importances(features, names, number):
    df_importance = pd.DataFrame({
        'Variable': names,
        'Importance': features
    })

    df_importance = df_importance.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Importance',
        y='Variable',
        data=df_importance.head(number),
        hue='Variable',
        palette='viridis',
        legend=False
    )

    plt.title(f"Top {number} des variables les plus importantes")
    plt.xlabel("Importance")
    plt.ylabel("Variables")
    plt.tight_layout()
    plt.show()
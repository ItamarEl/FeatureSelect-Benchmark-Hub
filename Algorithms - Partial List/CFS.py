import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler
import scipy.io
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from itertools import compress



import os
import numpy as np
import pandas as pd
import random as rn
from datetime import datetime
import warnings
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer


import os
import numpy as np
import pandas as pd
import random as rn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import entropy
import scipy.io

# Set seed for reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
rn.seed(seed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def su_calculation(f1, f2):
    """
    Calculate symmetrical uncertainty between two features, optimizing the use of entropy and mutual information.
    """
    joint_entropy = entropy(np.vstack((f1, f2)), base=2)
    entropy_f1 = entropy(f1, base=2)
    entropy_f2 = entropy(f2, base=2)
    mutual_info = entropy_f1 + entropy_f2 - joint_entropy
    return 2 * mutual_info / (entropy_f1 + entropy_f2) if (entropy_f1 + entropy_f2) > 0 else 0

def merit_calculation(X, y):
    """
    Efficiently calculates the merit of X given class labels y.
    """
    n_features = X.shape[1]
    rff = 0
    rcf = 0
    for i in range(n_features):
        rcf += su_calculation(X[:, i], y)
        for j in range(i + 1, n_features):
            rff += su_calculation(X[:, i], X[:, j])
    rff *= 2
    return rcf / np.sqrt(n_features + rff)

def cfs(X, y, k,row,seed):
    """
    Optimized CFS algorithm that selects up to k features based on their merits.
    """
    n_features = X.shape[1]
    selected_features = []
    feature_scores = []

    while len(selected_features) < k:
        best_feature = None
        best_merit = float('-inf')
        for i in range(n_features):
            if i not in selected_features:
                current_set = selected_features + [i]
                current_merit = merit_calculation(X[:, current_set], y)
                if current_merit > best_merit[]:
                    best_merit = current_merit
                    best_feature = i
        if best_feature is None:
            break
        selected_features.append(best_feature)
        feature_scores.append(best_merit)
        # Early stopping condition if no new features improve the merit
        if len(feature_scores) > 5 and all(x <= feature_scores[-1] for x in feature_scores[-5:]):
            break

    return selected_features, feature_scores



def run_cross_validation(train_data, train_labels, data_name, row, k,P,repository,algo_name, initial_splits, final_splits):
    results = []
    unique_values = np.unique(train_labels)
    best_accuracy = 0
    best_indices = None
    best_hyper_paramter = row['comment']
    start_time = datetime.now()

    label_arr = train_labels.copy()
    data_arr = train_data.copy()

    for n_splits in range(initial_splits, final_splits - 1, -1):
        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            accuracies, auc_scores_ovo, auc_scores_ovr = [], [], []
            CV_index = 0

            for train_index, test_index in skf.split(train_data, train_labels):
                fold_train_data, fold_test_data = train_data[train_index], train_data[test_index]
                fold_train_labels, fold_test_labels = train_labels[train_index], train_labels[test_index]


                # Feature selection based on current fold's training data
                indices_for_selected,feature_importances = cfs(fold_train_data, fold_train_labels,k,row,seed) 


                fold_train_data_selected = np.take(fold_train_data, indices_for_selected, axis=1)
                fold_test_data_selected = np.take(fold_test_data, indices_for_selected, axis=1)
                
                
                # Train model with current hyperparameters
                clf = ExtraTreesClassifier(n_estimators=200, max_depth=3, random_state=seed) # 200 and 3 are shown as the BIC hyperparameters and are also the default in scikit-learn.
                clf.fit(fold_train_data_selected, fold_train_labels)

                
                if len(unique_values) == 2:
                        fold_test_predicted_probs = clf.predict_proba(fold_test_data_selected)[:, 1]
                        fold_test_auc_ovo = round(roc_auc_score(fold_test_labels, fold_test_predicted_probs),3)
                        fold_test_auc_ovr = fold_test_auc_ovo.copy()
            
                else:
                        fold_test_predicted_probs = clf.predict_proba(fold_test_data_selected)
                        fold_test_auc_ovo = round(roc_auc_score(fold_test_labels, fold_test_predicted_probs, multi_class='ovo'),3)
                        fold_test_auc_ovr = round(roc_auc_score(fold_test_labels, fold_test_predicted_probs, multi_class='ovr'),3)
                 
                    
                accuracy = round(accuracy_score(fold_test_labels, clf.predict(fold_test_data_selected)),3)
                accuracies.append(accuracy)
                auc_scores_ovo.append(fold_test_auc_ovo)
                auc_scores_ovr.append(fold_test_auc_ovr)
                
                CV_index += 1
                print("CV Index: " + str(CV_index))
        
                # Update best indices if current accuracy is higher
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_indices = indices_for_selected.copy()
                    best_hyper_paramter = row['comment']


            running_time = datetime.now() - start_time
            running_time_Sec = round(running_time.total_seconds(), 1)

            results.append({
                'Data_Algo_Hyper_Key': f"{data_name}_{row['comment']}",
                'Repository': repository,
                'Dataset': data_name,
                'Algorithm': algo_name,
                'Hyperparamter': row['comment'],
                '#Folds': n_splits,
                '#successfuls_folds': CV_index,
                'Average Accuracy': round(np.mean(accuracies), 3),
                'Std_dev_Accuracy': round(np.std(accuracies), 3),
                'Max_Accuracy': round(np.max(accuracies), 3),
                'Accuracies': accuracies,
                "Average_auc_ovo": round(np.mean(auc_scores_ovo), 3),
                "Std_dev_auc_ovo": round(np.std(auc_scores_ovo), 3),
                "Max_auc_ovo": round(np.max(auc_scores_ovo), 3),
                'AUC_scores_ovo': auc_scores_ovo,
                "Average_auc_ovr": round(np.mean(auc_scores_ovr), 3),
                "Std_dev_auc_ovr": round(np.std(auc_scores_ovr), 3),
                "Max_auc_ovr": round(np.max(auc_scores_ovr), 3),
                'AUC_scores_ovr': auc_scores_ovr,
                'Running_time_5CV': running_time,
                'RunningTime_5CV[Sec]': running_time_Sec,
                'K_Feature_Importance_sorted_Hyperparamter': best_indices,
                'P': int(P),
                'N': len(train_labels),
                'P/N': round(int(P)/ len(train_labels), 3)
            })
            print(f"Finished with {n_splits}-fold CV")
            break  # Exit the loop if successful

        except Exception as e:
            print(f"Error with {n_splits}-fold CV: {e}")
            continue

    # If no successful CV, use whatever results are available
    if not results:
        running_time = datetime.now() - start_time
        running_time_Sec = round(running_time.total_seconds(), 1)
        results.append({
            'Data_Algo_Hyper_Key': f"{data_name}_{row['comment']}",
            'Repository': 'repository',
            'Dataset': data_name,
            'Algorithm': 'algo_name',
            'Hyperparamter': row['comment'],
            '#Folds': 'N/A',
            '#successfuls_folds': CV_index,
            'Average Accuracy': round(np.mean(accuracies), 3) if accuracies else 'N/A',
            'Std_dev_Accuracy': round(np.std(accuracies), 3) if accuracies else 'N/A',
            'Max_Accuracy': round(np.max(accuracies), 3) if accuracies else 'N/A',
            'Accuracies': accuracies,
            "Average_auc_ovo": round(np.mean(auc_scores_ovo), 3) if auc_scores_ovo else 'N/A',
            "Std_dev_auc_ovo": round(np.std(auc_scores_ovo), 3) if auc_scores_ovo else 'N/A',
            "Max_auc_ovo": round(np.max(auc_scores_ovo), 3) if auc_scores_ovo else 'N/A',
            'AUC_scores_ovo': auc_scores_ovo,
            "Average_auc_ovr": round(np.mean(auc_scores_ovr), 3) if auc_scores_ovr else 'N/A',
            "Std_dev_auc_ovr": round(np.std(auc_scores_ovr), 3) if auc_scores_ovr else 'N/A',
            "Max_auc_ovr": round(np.max(auc_scores_ovr), 3) if auc_scores_ovr else 'N/A',
            'AUC_scores_ovr': auc_scores_ovr,
            'Running_time_5CV': running_time,
            'RunningTime_5CV[Sec]': running_time_Sec,
            'K_Feature_Importance_sorted_Hyperparamter': best_indices,
            'P': int(P),
            'N': len(train_labels),
            'P/N': round(int(P)/ len(train_labels), 3)
        })

    return results, best_indices, best_accuracy, best_hyper_paramter

def alg(k, dataset_path, tempdf_hyperparamter, algo_name, repository, data_name, P,initial_splits, final_splits):
    # Load and preprocess data
    Data = scipy.io.loadmat(dataset_path)
    data_arr = Data['X']
    label_arr = Data['Y'][:, 0] if min(Data['Y'][:, 0]) == 0 else Data['Y'][:, 0] - 1

    # Normalize features
    Data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_arr)

    train_data = data_arr.copy()
    train_labels = label_arr.copy()

    results = []
    for index, row in tempdf_hyperparamter.iterrows():
        result, best_indices, best_accuracy, best_hyper_paramter = run_cross_validation(train_data, train_labels, data_name, row, k,P,repository,algo_name,initial_splits, final_splits)
        results.extend(result)

    df = pd.DataFrame(results)
    df['Best_Hyperparamter_Hypers'] = best_hyper_paramter
    df['Best_accuracy_Hypers'] = best_accuracy

    # Add the best_indices array to a new column for all rows in the DataFrame
    df['Best_K_Feature_Importance_sorted'] = [best_indices] * len(df)
    df['K'] = k

    return df


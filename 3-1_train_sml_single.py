from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import pandas as pd
import logging
from datetime import datetime

from Subject import Subject
import utils
import utils_score
from config import *


if __name__ == "__main__":
    # Create empty DataFrame with specific shape
    rows = []
    for DATA_TYPE in SML_TEST_DATA_TYPES_SINGLE:
        data_type_structure = DATA_TYPE[0]
        cols = DATA_TYPE[1]
        for model_name in SML_MODELS:
            rows.append(data_type_structure +"-"+ model_name)
    df = pd.DataFrame(index=rows, columns=cols)

    # for each data type
    for DATA_TYPE in SML_TEST_DATA_TYPES_SINGLE:
        # graph structure (topology)
        data_type_structure = DATA_TYPE[0]
        data_type_embbedings = DATA_TYPE[1]

        for metric in data_type_embbedings:
            # Set parameters
            key = data_type_structure +"-"+ metric

            # Set logger
            LOGGING_NAME = datetime.today().strftime("%Y%m%d-%H%M%S") +"_"+ data_type_structure +"_"+ metric
            logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)
            logger.info("Data type: {} | Metric: {}".format(data_type_structure, metric))
                    
            # load data
            subjects = utils.load_subjects(path_input=PATH, subject_pattern=SUBJECT_PATTERN, debug=False)
            if NUM_CLASSES==2:
                target = [int(subject.get_mstype(type="binary")) for subject in subjects]
            else:
                target = [int(subject.get_mstype()) for subject in subjects]
            logger.info("MS Types: {}".format(sorted(np.unique(target))))

            # for each model
            for model_name in SML_MODELS:
                # Set model name
                logger.info("+++ MODEL: {}".format(model_name))

                preds = np.zeros(len(subjects), dtype=float)

                # For each iteration
                glb_results = []

                for iteration in range(N_ITERATIONS):
                    logger.info("++++ Iteration: {} / {}".format(iteration+1, N_ITERATIONS))

                    # k-fold cross-validation
                    skf = StratifiedKFold(n_splits=N_SPLITS)
                    fold = 0
                    for train_index, test_index in skf.split(subjects, target):
                        fold += 1
                        logging.info("+++++ Fold: {}".format(fold))

                        # split dataset
                        X_train = [subjects[i] for i in train_index]
                        y_train = np.array(target)[train_index]
                        X_test  = [subjects[i] for i in test_index]
                        y_test  = np.array(target)[test_index]
                        
                        prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
                        prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
                        logger.info("Train set size     : {} [pwMS: {} - {:.2f} %]".format(len(X_train), y_train.sum(), prop_train*100))
                        logger.info("Test set size      : {} [pwMS: {} - {:.2f} %]".format(len(X_test), y_test.sum(), prop_test*100))

                        # get the length of the feature vector
                        len_feature_vector = X_train[0].get_metric(name=key).shape[1]
                        # list of data structures (one for each subject)
                        train_data = np.zeros(shape=(len(X_train), len_feature_vector), dtype=np.float64)
                        for i in range(len(X_train)):
                            train_data[i,:] = X_train[i].get_metric(name=key)
                            
                        test_data = np.zeros(shape=(len(X_test), len_feature_vector), dtype=np.float64)
                        for i in range(len(X_test)):
                            test_data[i,:] = X_test[i].get_metric(name=key)

                        # create the model
                        if model_name == "LogisticRegression":
                            model = LogisticRegression(max_iter=300)
                        elif model_name == "MLPClassifier":
                            model = MLPClassifier(hidden_layer_sizes=[200, 100, 10], activation='relu', solver='adam', random_state=1, batch_size=16, max_iter=300, learning_rate_init=0.001, early_stopping=True)
                        elif model_name == "RandomForestClassifier":
                            model = RandomForestClassifier()
                        elif model_name == "SupportVectorClassifier":
                            model = SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=1)
                        else:
                            raise ValueError("Model not supported: {}".format(model_name))
                        
                        # Train the model
                        model.fit(train_data, y_train)

                        # Predict the test set
                        y_pred = model.predict_proba(test_data)

                        # Save predictions (probabilities) to compute final test scores
                        preds[test_index] = y_pred[:,1]

                    # Compute metrics
                    auc_roc, _, _, _, _, _ = utils_score.compute_auc_binary(target=target, preds=preds, logger=logger)

                    # Plot ROC curve
                    if PLOTS:
                        utils_score.plot_auc_roc_binary(target=target, preds=preds, name_out=LOGGING_NAME+"_"+ model_name +"_"+ str(iteration+1) +"_ROC.png")

                    # Add to global results
                    glb_results.append(auc_roc)

                # Final AUC ROC score
                logger.info("AUC ROC scores: {}".format(glb_results))
                auc_roc_ave = np.mean(glb_results)
                auc_roc_std = np.std(glb_results)
                logger.info("*** Final AUC ROC score: {:.4f} +- {:.4f}".format(auc_roc_ave, auc_roc_std))

                # Define row nane to update score value in DF
                df_row_name = data_type_structure +"-"+ model_name
                # save metrics to DataFrame
                df.loc[df_row_name, metric] = "{:.4f} +- {:.4f}".format(auc_roc_ave, auc_roc_std)

    # Save DataFrame to CSV
    export_path = "results/"+ datetime.today().strftime("%Y%m%d-%H%M%S") +"_SML_Train_Single_Metrics.xlsx"
    df.to_excel(export_path, index=True)
    logger.info("DataFrame saved to: {}".format(export_path))
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_max_pool
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import numpy as np
import logging
from datetime import datetime

from Subject import Subject
from GCN import GCN
from GAT import GAT
from GIN import GIN
from ChebNet import ChebNet
from GraphSAGE import GraphSAGE
import utils
import utils_gnn
import utils_score
from config import *


if __name__ == "__main__":
    # create a list to store results
    global_results = []

    # for each data type
    for DATA_TYPE in TEST_DATA_TYPES:
        # name
        data_type_name = DATA_TYPE[0]
        # graph structure (topology)
        data_type_structure = DATA_TYPE[1]
        # embbedings
        data_type_embbedings = DATA_TYPE[2]

        # for each model architecture
        for model_arch in MODEL_ARCHITECTURES:
            MODEL = model_arch[0]
            # DIM_IN is the number of embeddings / node
            DIM_IN = len(METRIC_LIST) * len(data_type_embbedings)
            # DIM_OUT is the number of classes
            DIM_OUT = NUM_CLASSES
            # Hidden layers
            DIM_H1 = model_arch[1][0]
            DIM_H2 = model_arch[1][1]

            # for each Learning Rate and Weight Decay
            for LEARNING_RATE in LEARNING_RATES:
                # for each weight decay
                for WEIGHT_DECAY in WEIGHT_DECAYS:
                    # Set logger
                    EXP_NAME = data_type_name +"_"+ MODEL + "-"+ str(DIM_H1) + "-" + str(DIM_H2) + "_lr" + str(LEARNING_RATE) + "_wd" + str(WEIGHT_DECAY)
                    TIME_NAME = datetime.today().strftime("%Y%m%d-%H%M%S")
                    LOGGING_NAME = TIME_NAME +"_"+ EXP_NAME
                    logger = utils.set_logging(logging_level=logging.DEBUG, logging_name=LOGGING_NAME)
                    logger.info("Model: {} | Dim_in: {} | Dim_out: {} | Dim_h1: {} | Dim_h2: {} | LR: {} | WD: {}".format(MODEL, DIM_IN, DIM_OUT, DIM_H1, DIM_H2, LEARNING_RATE, WEIGHT_DECAY))
                    
                    # load data
                    subjects = utils.load_subjects(path_input=PATH, subject_pattern=SUBJECT_PATTERN, debug=DEBUG)
                    if NUM_CLASSES==2:
                        target = [int(subject.get_mstype(type="binary")) for subject in subjects]
                    else:
                        target = [int(subject.get_mstype()) for subject in subjects]
                    logger.info("MS Types: {}".format(sorted(np.unique(target))))
                    preds = np.zeros(len(subjects), dtype=int)

                    # For each iteration
                    glb_results = []

                    for iteration in range(N_ITERATIONS):
                        logger.info("++++ Iteration: {} / {}".format(iteration+1, N_ITERATIONS))

                        # k-fold cross-validation
                        skf = StratifiedKFold(n_splits=N_SPLITS)
                        fold = 0
                        for train_index, test_index in skf.split(subjects, target):
                            fold += 1
                            logging.info("Fold: {}".format(fold))

                            # split dataset
                            X_train = [subjects[i] for i in train_index]
                            X_test  = [subjects[i] for i in test_index]
                            y_train, y_test = np.array(target)[train_index], np.array(target)[test_index]
                            
                            prop_train = np.where(y_train == 1)[0].shape[0] / y_train.shape[0]
                            prop_test = np.where(y_test == 1)[0].shape[0] / y_test.shape[0]
                            logger.info("Train set size     : {}".format(len(X_train)))
                            logger.info("Test set size      : {}".format(len(X_test)))
                            logger.info("Train set % of pwMS: {:.4f} ({})".format(prop_train, y_train.sum()))
                            logger.info("Test set % of pwMS : {:.4f} ({})".format(prop_test, y_test.sum()))

                            # list of Data structures (one for each subject)
                            train_graphs = []
                            for i in range(len(X_train)):
                                g = utils_gnn.array_to_graph(subject=X_train[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                            num_classes=NUM_CLASSES, thr=THR, logger=logger)
                                train_graphs.append(g)
                                
                            test_graphs = []
                            for i in range(len(X_test)):
                                g = utils_gnn.array_to_graph(subject=X_test[i], data_type_structure=data_type_structure, data_type_embbedings=data_type_embbedings, 
                                                            num_classes=NUM_CLASSES, thr=THR, logger=logger)
                                test_graphs.append(g)

                            # create the model
                            if MODEL == "GCN":
                                model = GCN(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, logger=logger)
                            elif MODEL == "GAT":
                                model = GAT(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, dim_h2=DIM_H2, heads=NUM_HEADS, dropout=DROPOUT, lr=LEARNING_RATE, 
                                            wd=WEIGHT_DECAY, logger=logger)
                            elif MODEL == "GIN":
                                model = GIN(dim_in=DIM_IN, dim_out=DIM_OUT, hidden=DIM_H1, num_layers=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, logger=logger)
                            elif MODEL == "Cheb":
                                model = ChebNet(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, k=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, logger=logger)
                            elif MODEL == "SAGE":
                                model = GraphSAGE(dim_in=DIM_IN, dim_out=DIM_OUT, dim_h1=DIM_H1, num_layers=DIM_H2, lr=LEARNING_RATE, wd=WEIGHT_DECAY, logger=logger)
                            else:
                                logger.error("Model not implemented!")
                                raise NotImplementedError("Model not implemented!")
                            
                            model.fit(num_epocs=NUM_EPOCHS, train_graphs=train_graphs)
                            y_pred = model.test(test_graphs=test_graphs, y_test=y_test)
                            # Save predictions to compute final test scores
                            preds[test_index] = y_pred

                        if NUM_CLASSES == 2:
                            # Compute metrics for binary classification
                            auc_roc, _, _, _, _, _ = utils_score.compute_auc_binary(target=target, preds=preds, logger=logger)
                            if PLOTS:
                                utils_score.plot_auc_roc_binary(target=target, preds=preds, name_out=LOGGING_NAME +"_"+ str(iteration+1))
                        else:
                            # Compute metrics
                            utils_score.compute_auc_multiclass(target=target, preds=preds, logger=logger)
                            if PLOTS:
                                utils_score.plot_auc_roc_multiclass(target=target, preds=preds, name_out=LOGGING_NAME +"_"+ str(iteration+1))
                        
                        # Add to global results
                        glb_results.append(auc_roc)

                    # Final AUC ROC score
                    logger.info("AUC ROC scores: {}".format(glb_results))
                    auc_roc_ave = np.mean(glb_results)
                    auc_roc_std = np.std(glb_results)
                    logger.info("*** Final AUC ROC score: {:.4f} +- {:.4f}".format(auc_roc_ave, auc_roc_std))

                    # Store to results
                    global_results.append({"NAME": EXP_NAME,
                                           "Model": MODEL,
                                           "Dim_in": DIM_IN,
                                           "Dim_out": DIM_OUT,
                                           "Dim_h1": DIM_H1,
                                           "Dim_h2": DIM_H2,
                                           "Learning_rate": LEARNING_RATE,
                                           "Weight_decay": WEIGHT_DECAY,
                                           "AUC_ROC_AVE": auc_roc_ave,
                                           "AUC_ROC_STD": auc_roc_std})
    # Export to TXT file
    with open("results/"+ TIME_NAME +"_GNNs.txt", "w") as f:
        for line in global_results:
            f.write("{} : AUC ROC = {:.4f} +- {:.4f} \n".format(line["NAME"], line["AUC_ROC_AVE"], line["AUC_ROC_STD"]))

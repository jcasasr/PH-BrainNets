import os
import pickle
import numpy as np
import networkx as nx
from typing import List
import logging
from datetime import datetime
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import BettiCurve
from gtda.plotting import plot_diagram

from Subject import Subject
import utils_metrics


def check_matrix(matrix_name:str, matrix:np.ndarray, min_value:float=0, max_value:float=1, data_type:str="SL") -> bool:
    # Check size
    if data_type == "SL":
        assert matrix.shape == (76,76), "Incorrect shape {} in matrix {}!".format(matrix.shape, matrix_name)
    else:
        assert matrix.shape == (152,152), "Incorrect shape {} in matrix {}!".format(matrix.shape, matrix_name)
    # Check values
    assert np.min(matrix) >= min_value, "Values < {} in matrix {}!".format(min_value, matrix_name)
    assert np.max(matrix) <= max_value, "Values > {} in matrix {}!".format(max_value, matrix_name)
    # Check diagonal
    assert np.all(matrix.diagonal() == 0) == True, "Diagonal values > 0 in matrix {}!".format(matrix_name)

    return True


def check_subject(subject:Subject):
    # Check SL matrices
    matrix_names = ['FA', 'GM', 'RS']

    for matrix_name in matrix_names:
        matrix = subject.get_matrix(matrix_name)
        if matrix is not None:
            check_matrix(matrix_name=matrix_name, matrix=matrix, min_value=0, max_value=1, data_type="SL")
        else:
            raise AssertionError("Matrix not found!")
    
    # Check ML matrix
    matrix = subject.get_matrix("ML")
    if matrix is not None:
        check_matrix(matrix_name="ML", matrix=matrix, min_value=0, max_value=1, data_type="ML")
    else:
        raise AssertionError("Matrix not found!")

    print("Subject {} checked!".format(subject.get_ID()))


def create_multilayer(FA, RS, GM):
    num_nodes = FA.shape[0]
    
    # data structure
    ml = np.zeros((num_nodes*2, num_nodes*2), dtype=float)
    ml[:76,:76] = np.abs(RS[:,:])
    ml[76:,76:] = np.abs(GM[:,:])
    ml[76:,:76] = np.abs(FA[:,:])
    ml[:76,76:] = np.abs(FA[:,:])
        
    return ml


def compute_ml_metrics(subject: Subject, logger=None):
    """
    Compute metrics for a subject
    :param subject: Subject object
    :param logger: Logger object
    :return: Subject object with metrics
    """
    # Params
    data_type = "ML"
    metric_list = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality']

    # get data 
    A = subject.get_matrix(data_type)
    ID = subject.get_ID()
    bool_reshape = True

    logger.debug("Computing {} metrics for subject {}...".format(data_type, ID))
    
    for metric in metric_list:
        # compute the specific metric
        logger.debug("Computing {}...".format(metric))

        try:
            if metric=='Degree':
                temp = np.count_nonzero(A > 0, axis=0)

            elif metric=='Strength':
                temp = np.sum(A, axis=0)

            elif metric=='LocalEfficiency':
                # Use 'distance'
                A_inv = utils_metrics.create_distance_A_from_A(A)

                # Compute A min
                A_min = utils_metrics.compute_A_min(A_inv)

                # create G
                G = utils_metrics.create_graph_from_AM(A_min)

                # compute LE as a single layer
                temp = utils_metrics.compute_LE_SL(G)

                # Do not reshape 
                bool_reshape = False

            elif metric=='ClosenessCentrality':
                # Use 'distance'
                A_inv = utils_metrics.create_distance_A_from_A(A)

                # Compute A min
                A_min = utils_metrics.compute_A_min(A_inv)

                # create G
                G = utils_metrics.create_graph_from_AM(A_min)

                # compute LE as a single layer
                temp = np.array(list(nx.closeness_centrality(G, distance='weight').values()))
                
                # Do not reshape 
                bool_reshape = False

            elif metric=='BetweennessCentrality':
                # Use 'distance'
                A_inv = utils_metrics.create_distance_A_from_A(A)

                # Compute A min
                A_min = utils_metrics.compute_A_min(A_inv)

                # create G
                G = utils_metrics.create_graph_from_AM(A_min)

                # compute LE as a single layer
                temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values()))
                
                # Do not reshape 
                bool_reshape = False
            
            else:
                raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(metric))
        except Exception as e:
                print(e)
                raise e
    
        # Folding results to get 76 nodes (instead of 152)
        if bool_reshape:
            # reshape and sum values
            temp2 = temp.reshape((76,2), order='F')
            temp3 = np.sum(temp2, axis=1)
        else:
            temp3 = temp
        
        # store the results
        key = data_type +"-"+ metric
        subject.set_metric(key, temp3)
        logger.debug("Results of metric {} on {} stored using key='{}'".format(metric, data_type, key))
        
    return subject


def compute_sl_metrics(subject: Subject, logger=None):
    """
    Compute metrics for a subject
    :param subject: Subject object
    :param logger: Logger object
    :return: Subject object with metrics
    """
    # Params
    data_types = ["FA", "GM", "RS"]
    metric_list = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality']

    # get data type (i.e. matrix name)
    for data_type in data_types:
        # get data 
        A = subject.get_matrix(data_type)
        ID = subject.get_ID()

        logger.debug("Computing {} metrics for subject {}...".format(data_type, ID))
        
        for metric in metric_list:
            # compute the specific metric
            logger.debug("Computing {}...".format(metric))

            # create graph
            G = utils_metrics.create_graph_from_AM(A)

            # compute metric values
            try:
                if metric=='Degree':
                    temp = np.array(list(nx.degree_centrality(G).values()))

                elif metric=='Strength':
                    temp = np.sum(A, axis=0)

                elif metric=='Clustering':
                    temp = np.array(list(nx.clustering(G, weight='weight').values()))

                elif metric=='BetweennessCentrality':
                    # Use 'distance'
                    G = utils_metrics.create_distance_graph_from_AM(A)

                    # debug only
                    utils_metrics.report_graph_basics(G)

                    temp = np.array(list(nx.betweenness_centrality(G, k=None, normalized=True, weight='weight').values()))

                elif metric=='ClosenessCentrality':
                    # Use 'distance'
                    G = utils_metrics.create_distance_graph_from_AM(A)

                    temp = np.array(list(nx.closeness_centrality(G, distance='weight').values()))

                elif metric=='EigenvectorCentrality':
                    temp = np.array(list(nx.eigenvector_centrality(G, max_iter=100, tol=1e-06, weight='weight').values()))

                elif metric=='PageRank':
                    temp = np.array(list(nx.pagerank(G, max_iter=1000, weight='weight').values()))

                elif metric=='LocalEfficiency':
                    # Use 'distance'
                    G = utils_metrics.create_distance_graph_from_AM(A)

                    # Implementation of LE on single layer networks
                    temp = utils_metrics.compute_LE_SL(G)
                
                else:
                    raise Exception("ERROR: Incorrect metric value! (METRIC is {})".format(metric))
            except Exception as e:
                    print(e)
                    raise e

            # store the results
            key = data_type +"-"+ metric
            subject.set_metric(key, temp)
            logger.debug("Results of metric {} on {} stored using key='{}'".format(metric, data_type, key))

    return subject


def compute_ph_metric(subject: Subject, logger=None):
    """
    Compute PH for a subject
    :param subject: Subject object
    :param logger: Logger object
    :return: Subject object with metrics
    """
    # Params
    data_types = ["FA", "GM", "RS", "ML"]
    homology_dimensions_list = [[0], [0, 1], [0, 1, 2]] # 0; 0 and 1; and 0, 1 and 2 dimensions

    # get data type (i.e. matrix name)
    for data_type in data_types:
        # get data 
        A = subject.get_matrix(data_type)
        ID = subject.get_ID()

        for homology_dimensions in homology_dimensions_list:
            logger.debug("Computing {} PH metric for homology dimensions {} for subject {}...".format(data_type, homology_dimensions, ID))

            # compute metric values
            try:
                A = 1 - A
                np.fill_diagonal(A, 0)
                VR = VietorisRipsPersistence(metric="precomputed", homology_dimensions=homology_dimensions, n_jobs=-1)
                diagrams = VR.fit_transform([A])

                b = BettiCurve(n_bins=100) # Compute the Betti curve from the persistence diagrams
                temp = b.fit_transform(diagrams) # returns: ndarray of shape (n_samples, n_homology_dimensions, n_bins=100 (default))
                temp = temp.reshape(-1) # as a flat array
            except Exception as e:
                    print(e)
                    raise e

            # store the results
            key = data_type +"-PH_" + "-".join(map(str, homology_dimensions))
            subject.set_metric(key, temp)
            logger.debug("Results of {} PH metric for homology dimensions {} stored using key='{}'".format(data_type, homology_dimensions, key))

    return subject
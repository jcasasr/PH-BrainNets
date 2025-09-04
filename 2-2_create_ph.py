import os
import pickle
import numpy as np
from typing import List
import logging
from sklearn.preprocessing import StandardScaler

from Subject import Subject
import utils
import utils_graphs
from config import *


if __name__ == "__main__":
    logger = utils.set_logging(logging_level=logging.DEBUG)
    num_subjs = 0

    files = sorted(os.listdir(path=PATH))
    for file in files:
        if file.endswith(".pkl"):
            subject = utils.load_object(os.path.join(PATH, file))
            if DEBUG:
                logger.debug("\tProcessing subject {}...".format(subject.get_ID()))

            # Compute graph metrics
            subject = utils_graphs.compute_ph_metric(subject, logger=logger) # PH

            # Export subject
            out_filename = os.path.join(PATH, subject.get_ID() +".pkl")
            utils.save_object(subject, out_filename)
            num_subjs += 1

    logger.info("... Process finished!")

    logger.info("Data standarization process...")

    # Params
    data_types = ["FA", "GM", "RS", "ML"]
    homology_dimensions_list = [[0], [0, 1], [0, 1, 2]] # 0; 0 and 1; and 0, 1 and 2 dimensions

    for data_type in data_types:
        for homology_dimensions in homology_dimensions_list:
            logger.info("Normalizing {} PH metric for homology dimensions {}...".format(data_type, homology_dimensions))

            # Variables
            X = np.zeros((num_subjs, 100 * len(homology_dimensions)), dtype=float) # Homology n * 100 bins
            i_subj = 0

            for file in files:
                if file.endswith(".pkl"):
                    subject = utils.load_object(os.path.join(PATH, file))
                    if DEBUG:
                        logger.debug("\tProcessing subject {}...".format(subject.get_ID()))

                    key = data_type +"-PH_" + "-".join(map(str, homology_dimensions))
                    X[i_subj, :] = subject.get_metric(key)
                    i_subj += 1

            # Normalize the data
            logger.info("Computing 'StandardScaler'...")
            scaler = StandardScaler()
            scaler.fit(X)

            logger.info("Updating data...")
            for file in files:
                if file.endswith(".pkl"):
                    subject = utils.load_object(os.path.join(PATH, file))
                    if DEBUG:
                        logger.debug("\tExporting subject {}...".format(subject.get_ID()))

                    # Update data
                    data = scaler.transform(subject.get_metric(key).reshape(1, -1))
                    subject.set_metric(key, data)

                    # Export subject
                    out_filename = os.path.join(PATH, subject.get_ID() +".pkl")
                    utils.save_object(subject, out_filename)

            logger.info("... Process finished!")

    logger.info("Process finished!")
import os
import pickle
import numpy as np
from typing import List
import logging

from Subject import Subject
import utils
import utils_graphs
from config import *


if __name__ == "__main__":
    logger = utils.set_logging(logging_level=logging.DEBUG)

    files = sorted(os.listdir(path=PATH))
    for file in files:
        if file.endswith(".pkl"):
            subject = utils.load_object(os.path.join(PATH_1, file))
            if DEBUG:
                logger.info("\tProcessing subject {}...".format(subject.get_ID()))

            # Create multilayer
            FA = subject.get_matrix("FA")
            GM = subject.get_matrix("GM")
            RS = subject.get_matrix("RS")
            ml = utils_graphs.create_multilayer(FA, RS, GM)
            subject.set_matrix(name="ML", value=ml)
    
            # check subject
            utils_graphs.check_subject(subject)

            # Compute graph metrics
            subject = utils_graphs.compute_ml_metrics(subject, logger=logger) # ML
            subject = utils_graphs.compute_sl_metrics(subject, logger=logger) # FA, GM, RS

            # Export subject
            out_filename = os.path.join(PATH, subject.get_ID() +".pkl")
            utils.save_object(subject, out_filename)

    logger.info("Process finished!")

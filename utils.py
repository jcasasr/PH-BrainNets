import os
import pickle
from typing import List
import logging
from datetime import datetime
import fnmatch
import os
import numpy as np
import pandas as pd

from Subject import Subject


def set_logging(logging_level=logging.DEBUG, logging_name=None):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # remove older loggers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    # Create a formatter to define the log format
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    # Create a file handler to write logs to a file
    if logging_name is None:
        logging_name = datetime.today().strftime("%Y%m%d-%H%M%S")

    file_handler = logging.FileHandler(filename=os.path.join("logs", logging_name + "_log.txt"))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a stream handler to print logs to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def save_object(obj, filename):
    # Overwrites any existing file
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    print("   Subject {} exported!".format(obj.get_ID()))


def load_object(filename, debug=False):
    # Load the object from the file
    with open(filename, 'rb') as inp:
        objname = pickle.load(inp)
        if debug:
            print("Loading subject {}...".format(objname.get_ID()))

        return objname
    

def load_subjects(path_input:str="", subject_pattern:str="*.pkl", debug=False) -> List:
    files = sorted(os.listdir(path=path_input))
    subjects = []
    for file in files:
        if fnmatch.fnmatch(file, subject_pattern):
            subject = load_object(os.path.join(path_input, file), debug=debug)
            subjects.append(subject)
            if debug:
                print("Subject {} loaded!".format(subject.get_ID()))

    return subjects


def create_subjects(path_input:str="", path_output:str="", debug=False) -> List:
    """
    Create subjects from the input data.
    :param path_input: Path to the input data
    :param path_output: Path to the output data
    :param debug: Debug mode
    :return: List of subjects
    """
    # Load the data
    info = pd.read_csv(os.path.join(path_input, "ID_info.csv"))
    data_FA = np.load(os.path.join(path_input, "data", 'data_FA.npy'))
    data_GM = np.load(os.path.join(path_input, "data", 'data_GM.npy'))
    data_RS = np.load(os.path.join(path_input, "data", 'data_RS.npy'))

    # atts
    atts = ["gender", "edss", "dobirth", "doscan", "dostart", "age", "DD"]

    # List of subjects
    subjects = []

    for i in range(len(info)):
        print("Processing subject {}...".format(i))

        # Get the information
        ID = info.loc[i, "ID"]
        cohort = info.loc[i, "origin"]
        ID_old = info.loc[i, "ID_old"]
        ms_type = info.loc[i, "mstype"]

        # change the ID to "cohort-ID"
        ID = cohort + "-" + "{:04}".format(ID)

        # Create Subject object
        subject = Subject(ID, cohort, ID_old, ms_type)

        # Set attributes
        for att in atts:
            subject.set_attribute(att, info.loc[i, att])

        # Set diagonal to 0
        tmp_data_FA = set_diagonal(matrix=data_FA[i], value=0)
        tmp_data_GM = set_diagonal(matrix=data_GM[i], value=0)
        tmp_data_RS = set_diagonal(matrix=data_RS[i], value=0)
        
        # Set matrices
        subject.set_matrix("FA", tmp_data_FA)
        subject.set_matrix("GM", np.abs(tmp_data_GM))
        subject.set_matrix("RS", np.abs(tmp_data_RS))

        # Export subject
        out_filename = os.path.join(path_output, ID +".pkl")
        save_object(subject, out_filename)

        # Append to the list
        subjects.append(subject)

        if debug:
            print(subject)

    return subjects


def set_diagonal(matrix, value:float=0):
    """
    Set the diagonal of a matrix to a specific value.
    :param matrix: Matrix to set the diagonal
    :param value: Value to set the diagonal
    :return: Matrix with the diagonal set
    """
    np.fill_diagonal(matrix, value)
    return matrix
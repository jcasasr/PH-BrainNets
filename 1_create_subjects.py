import numpy as np
import pandas as pd
import os
import pickle
from typing import List

from Subject import Subject
from utils import *
from config import *


if __name__ == "__main__":
    # Load the subjects from BCN
    subjects = create_subjects(PATH_BCN, PATH, debug=True)
    print("Number of subjects: ", len(subjects))

    # Load the subjects from NAP
    subjects = create_subjects(PATH_NAP, PATH, debug=True)
    print("Number of subjects: ", len(subjects))
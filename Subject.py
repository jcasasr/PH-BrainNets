from enum import Enum
from typing import Dict 


class MSType(Enum):
    HV = 0
    RRMS = 1
    SPMS = 2
    PPMS = 3


class Subject():
    _ID: str
    _cohort: str
    _ID_old: str
    _mstype: MSType
    _attributes: Dict
    _matrices: Dict
    _metrics: Dict

    def __init__(self, ID:str, cohort:str, ID_old:str, mstype:MSType) -> None:
        self._ID = ID
        self._cohort = cohort
        self._ID_old = ID_old
        self._mstype = mstype
        self._attributes = {}
        self._matrices = {}
        self._metrics = {}

    ### Getters
    def get_ID(self) -> str:
        return self._ID

    def get_cohort(self) -> str:
        return self._cohort

    def get_ID_old(self) -> str:
        return self._ID_old

    def get_mstype(self, type:str="multiclass") -> MSType:
        # Stored: -1: HV, 0: RRMS, 1: SPMS, 2: PPMS
        # Return: 0: HV, 1: RRMS, 2: SPMS, 3: PPMS
        if type=="binary":
            return 0 if self._mstype==-1 else 1
        else:
            return self._mstype + 1 

    ### Attributes
    def get_attribute(self, name:str="") -> str:
        if self._attributes[name] is not None:
            return self._attributes[name]
        else:
            return None

    def set_attribute(self, name:str, value:str) -> bool:
        self._attributes[name] = value
        return True

    ### Matrices
    def get_matrix(self, name:str=""):
        if self._matrices[name ]is not None:
            return self._matrices[name]
        else:
            return None

    def set_matrix(self, name:str, value) -> bool:
        self._matrices[name] = value
        return True

    ### Metrics
    def get_metric(self, name:str=""):
        if self._metrics[name] is not None:
            return self._metrics[name]
        else:
            return None

    def set_metric(self, name:str, value:float) -> bool:
        self._metrics[name] = value
        return True

    def __str__(self) -> str:
        return "Subject: {} --> Cohort: {} and MS type: {} [Old ID: {}]".format(self._ID, self._cohort, self._mstype, self._ID_old)
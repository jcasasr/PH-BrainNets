##################################
### Paths and general settings ###
##################################
DEBUG = True
PLOTS = False
PATH_BCN = "/Users/jcasasr/Library/CloudStorage/Dropbox/CVC_MRI-Data/BCN_output"
PATH_NAP = "/Users/jcasasr/Library/CloudStorage/Dropbox/CVC_MRI-Data/NAP_output"
PATH = "./data"
MATRIX_TYPES = ["FA", "GM", "RS", "ML"] # Types of valid matrices
N_ITERATIONS = 4

####################
### Data sources ###
####################
# All: "*.pkl" | BCN: "BCN-*.pkl" | NAP: "NAP-*.pkl"
SUBJECT_PATTERN = "NAP-*.pkl"
# Data models
DATA_TYPES = ["FA", "GM", "RS", "ML"]
# List of metrics to be computed
METRIC_LIST = ['Degree', 'Strength', 'LocalEfficiency', 'ClosenessCentrality', 'BetweennessCentrality']

######################
### PIPELINE 1 & 2:
# PH & Graph + Supervised ML settings 
######################

### Single Modality Learning (SML) settings
# Data types to be tested
# Tuple: (graph_structure, [embeddings])
SML_TEST_DATA_TYPES_SINGLE = [("FA", ["Degree", "Strength", "LocalEfficiency", "ClosenessCentrality", "BetweennessCentrality"]), 
                       ("GM", ["Degree", "Strength", "LocalEfficiency", "ClosenessCentrality", "BetweennessCentrality"]), 
                       ("RS", ["Degree", "Strength", "LocalEfficiency", "ClosenessCentrality", "BetweennessCentrality"]), 
                       ("ML", ["Degree", "Strength", "LocalEfficiency", "ClosenessCentrality", "BetweennessCentrality"]),
                       ("FA", ["PH_0", "PH_0-1", "PH_0-1-2"]), 
                       ("GM", ["PH_0", "PH_0-1", "PH_0-1-2"]), 
                       ("RS", ["PH_0", "PH_0-1", "PH_0-1-2"]), 
                       ("ML", ["PH_0", "PH_0-1", "PH_0-1-2"]),
                       ]

### Multi Modality Learning (SML) settings
# Data types to be tested
# Tuple: (graph_structure, [embeddings - All features must be prefixed with the graph structure and will be concatenated into a single feature vector])
SML_TEST_DATA_TYPES_MULTI = [("MultiModal-Degree", ["FA-Degree", "GM-Degree", "RS-Degree"]), 
                       ("MultiModal-Strength", ["FA-Strength", "GM-Strength", "RS-Strength"]),
                       ("MultiModal-LocalEfficiency", ["FA-LocalEfficiency", "GM-LocalEfficiency", "RS-LocalEfficiency"]),
                       ("MultiModal-ClosenessCentrality", ["FA-ClosenessCentrality", "GM-ClosenessCentrality", "RS-ClosenessCentrality"]),
                       ("MultiModal-BetweennessCentrality", ["FA-BetweennessCentrality", "GM-BetweennessCentrality", "RS-BetweennessCentrality"]),
                       ("MultiFeature-FA", ["FA-Degree", "FA-Strength", "FA-LocalEfficiency", "FA-ClosenessCentrality", "FA-BetweennessCentrality"]),
                       ("MultiFeature-GM", ["GM-Degree", "GM-Strength", "GM-LocalEfficiency", "GM-ClosenessCentrality", "GM-BetweennessCentrality"]),
                       ("MultiFeature-RS", ["RS-Degree", "RS-Strength", "RS-LocalEfficiency", "RS-ClosenessCentrality", "RS-BetweennessCentrality"]),
                       ("MultiModal-PH_0", ["FA-PH_0", "GM-PH_0", "RS-PH_0"]),
                       ("MultiModal-PH_0-1", ["FA-PH_0-1", "GM-PH_0-1", "RS-PH_0-1"]),
                       ("MultiModal-PH_0-1-2", ["FA-PH_0-1-2", "GM-PH_0-1-2", "RS-PH_0-1-2"]),
                       ]

SML_MODELS = ['RandomForestClassifier', 'MLPClassifier', 'LogisticRegression', 'SupportVectorClassifier']

######################
### PIPELINE 3:
# GNN settings 
######################
# Data types to be tested
# Tuple: (name, graph_structure, [embeddings])
TEST_DATA_TYPES = [("FA", "FA", ["FA"]), 
                   ("GM", "GM", ["GM"]), 
                   ("RS", "RS", ["RS"]), 
                   ("ML", "ML", ["ML"]), 
                   ("MM", "FA", ["GM", "RS"])]

# Number of classes:
# binary: 2 
# multiclass: 4
NUM_CLASSES = 2
## PARAMS
NUM_EPOCHS = 100
N_SPLITS = 4
THR = 0.4
## Model architecture
# Tuple: (model, [dim_h1, dim_h2])
# GCN: dim_h1=hidden channels 1, dim_h2=hidden channels 2 (0 if only one conv layer)
# GAT: dim_h1=hidden channels 1, dim_h2=hidden channels 2 (0 if only one conv layer)
# GIN: dim_h1=hidden, dim_h2=num_layers
# Cheb: dim_h1=hidden channels, dim_h2=k (Chebyshev filter size)
# SAGE: dim_h1=hidden, dim_h2=num_layers
MODEL_ARCHITECTURES = [("GCN", [32, 0]),
                       ("GCN", [32, 16]),
                       ("GAT", [32, 0]),
                       ("GAT", [32, 16]),
                       ("GIN", [32, 3]),
                       ("GIN", [32, 5]),
                       ("Cheb", [32, 3]),
                       ("Cheb", [32, 5]),
                       ("SAGE", [32, 3]),
                       ("SAGE", [32, 5])]
# Learning rates and weight decays to be tested
LEARNING_RATES = [0.01, 0.001, 0.0001]
WEIGHT_DECAYS = [5e-4, 5e-6]
# GAT params
NUM_HEADS = 4
DROPOUT = 0.6
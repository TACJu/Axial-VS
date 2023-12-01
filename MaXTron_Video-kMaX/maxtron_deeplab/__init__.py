from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maxtron_deeplab_config

# dataset loading
from .data.dataset_mappers.vipseg_panoptic_maxtron_dataset_mapper import VIPSegPanopticMaXTronDatasetMapper

# models
from .maxtron_wc_model import MaXTronWCDeepLab
from .maxtron_cc_model import MaXTronCCDeepLab

# evaluation
from .evaluation.vipseg_evaluation import VIPSegEvaluator

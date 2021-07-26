# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Modification
# - Import functions/classes relevant to GLUE and Open-domain QA experiments using BERTAC

from .glue import glue_convert_examples_to_features, glue_output_modes, glue_processors, glue_tasks_num_labels
from .openqa import OpenQAExample, OpenQAFeatures, OpenQAV1Processor, OpenQAV2Processor, openqa_convert_examples_to_features
from .utils import DataProcessor, InputExample, InputFeatures, InputFeatures4GLUE, InputExample4GLUE, SingleSentenceClassificationProcessor

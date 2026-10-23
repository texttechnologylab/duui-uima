"""
Simple python script, that just creates an instance of the pipline class what leads to the actual model getting downloaded.
This can be started in the Dockerfile so the model is already downloaded when using the image.
"""

import logging
import duui_logging
from duui_logging import log_info, log_warn, log_error, log_debug

from transformers import pipeline
pipeline_classification_topics = pipeline("text-classification", model="chkla/parlbert-topic-german")
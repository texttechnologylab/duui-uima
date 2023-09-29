"""
Simple python script, that just creates an instance of the pipline class what leads to the actual model getting downloaded.
This can be started in the Dockerfile so the model is already downloaded when using the image.
"""

from transformers import pipeline
pipeline_classification_topics = pipeline("text-classification", model="chkla/parlbert-topic-german")
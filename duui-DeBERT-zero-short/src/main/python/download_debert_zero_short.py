"""
Simple python script, that just creates an instance of the pipline class what leads to the actual model getting downloaded.
This can be started in the Dockerfile so the model is already downloaded when using the image.
"""

from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
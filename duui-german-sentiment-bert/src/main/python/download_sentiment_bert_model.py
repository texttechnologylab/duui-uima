"""
Simple python script, that just creates an instance of the SentimentModel class what leads to the actual model getting downloaded.
This can be started in the Dockerfile so the model is already downloaded when using the image.
"""

from germansentiment import SentimentModel
model = SentimentModel()

SUPPORTED_MODEL = {
    "LiYuan/amazon-review-sentiment-analysis": {
        "version": "0aacda6423e43213da4e50a0f30cfcdb42a5c725",
        "max_length": 512,
        "mapping": {
            "5 stars": 1,
            "4 stars": 0.5,
            "3 stars": 0,
            "2 stars": -0.5,
            "1 star": -1
        },
        "3sentiment": {
            "pos": ["5 stars", "4 stars"],
            "neu": ["3 stars"],
            "neg": ["2 stars", "1 star"]
        },
        "preprocess": lambda text: text.lower(),
        "languages": ["en", "de", "fr", "es", "it", "nl"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)

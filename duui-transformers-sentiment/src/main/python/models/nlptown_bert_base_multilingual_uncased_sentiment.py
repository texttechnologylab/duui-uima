SUPPORTED_MODEL = {
    "nlptown/bert-base-multilingual-uncased-sentiment": {
        "version": "e06857fdb0325a7798a8fc361b417dfeec3a3b98",
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

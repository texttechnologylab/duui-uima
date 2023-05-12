SUPPORTED_MODEL = {
    "cmarkea/distilcamembert-base-sentiment": {
        "version": "b7804e295dc3cf2aa8ce8cff83f22e0bdd249558",
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
        "preprocess": lambda text: text,
        "languages": ["fr"]
    },
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)

# See "TweetEval" paper sec. 2.2 and https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
def cardiffnlp_preprocess(text: str) -> str:
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


SUPPORTED_MODEL = {
    "cardiffnlp/twitter-roberta-base-sentiment": {
        "version": "b636d90b2ed53d7ba6006cefd76f29cd354dd9da",
        "max_length": 512,
        "mapping": {
            "LABEL_2": 1,
            "LABEL_1": 0,
            "LABEL_0": -1
        },
        "3sentiment": {
            "pos": ["LABEL_2"],
            "neu": ["LABEL_1"],
            "neg": ["LABEL_0"]
        },
        "preprocess": lambda text: cardiffnlp_preprocess(text),
        "languages": ["en"]
    }
}

if __name__ == "__main__":
    import preload_models
    preload_models.preload(SUPPORTED_MODEL)

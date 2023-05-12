if __name__ != "__main__":
    from .oliverguhr_german_sentiment_bert import oliverguhr_clean_text

SUPPORTED_MODEL = {
    "dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-2-unseen-adapter-pfeiffer-ep10-cp126710": {
        "version": "v1.0_ep10_cp126710",
        "type": "adapter",
        "model_type": "huggingface",
        "model_name": "oliverguhr/german-sentiment-bert",
        "model_version": "c5c8dd0c5b966460dce1b7c5851bd90af1d2c6b6",
        "adapter_type": "local",
        "adapter_path": "/models/dbaumartz/oliverguhr_german-sentiment-bert-finetuned-de-3sentiment-unseen-adapter-pfeiffer/checkpoint-126710/3sentiment-unseen",
        "max_length": 512,
        "mapping": {
            "positive": 1,
            "neutral": 0,
            "negative": -1
        },
        "3sentiment": {
            "pos": ["positive"],
            "neu": ["neutral"],
            "neg": ["negative"]
        },
        "preprocess": lambda text: oliverguhr_clean_text(text),
        "languages": ["de"]
    },
}

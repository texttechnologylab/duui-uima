from transformers import pipeline


def preload(models):
    for model_name in models:
        model_revision = models[model_name]["version"]
        print("loading", model_name, model_revision)
        pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            revision=model_revision
        )

from transformers import pipeline


def preload(models):
    for model_name in models:
        model_type = "huggingface" if not "type" in models[model_name] else models[model_name]["type"]
        if model_type == "huggingface":
            model_revision = models[model_name]["version"]
            print("loading", model_name, model_revision)
            pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                revision=model_revision
            )
        else:
            print("model not in huggingface hub, please add to docker manually!")

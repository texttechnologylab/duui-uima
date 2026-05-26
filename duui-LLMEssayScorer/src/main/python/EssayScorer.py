from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
from openai import OpenAI
from scipy.special import softmax
from langchain_core.runnables import RunnableLambda
model_name_list = {
"KevSun/Engessay_grading_ML": ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"],
"JacobLinCool/IELTS_essay_scoring_safetensors": ["Task Achievement", "Coherence and Cohesion", "Vocabulary", "Grammar", "Overall"]
}

class EssayScorer:
    def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.device = torch.device(device)
        self.model_name = model_name
        self.model.eval()

    def run_messages(self, texts: list[str]) -> list[dict[str, float]]:
        with torch.no_grad():
            score_list = []
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            outputs = self.model(**inputs)
            predictions = outputs.logits.squeeze()
            predicted_scores = predictions.detach().cpu().numpy()
            item_names = model_name_list[self.model_name]
            match self.model_name:
                case "KevSun/Engessay_grading_ML":
                    # Scale predictions from 1 to 10 and round to the nearest 0.5
                    scaled_scores = 2.25 * predicted_scores - 1.25
                    rounded_scores = np.round(scaled_scores * 2) / 2# Round to nearest 0.5
                case "JacobLinCool/IELTS_essay_scoring_safetensors":
                    normalized_scores = (predicted_scores / predicted_scores.max()) * 9  # Scale to 9
                    rounded_scores = np.round(normalized_scores * 2) / 2
                case _:
                    raise ValueError(f"Model {self.model_name} is not supported.")
            # if rounded_scores.ndim == 1: format to 2dim
            if rounded_scores.ndim == 1:
                rounded_scores = rounded_scores.reshape(1, -1)
            for scores in rounded_scores:
                score_dict = {}
                for item, score in zip(item_names, scores):
                    score_dict[item] = float(score)
                score_list.append(score_dict)
        return score_list

# class EssayScorerPip:
#     def __init__(self, model_name, device="cuda" if torch.cuda.is_available() else "cpu"):
#         self.model_name = model_name
#
#     def process_texts(self, texts: list[str]) -> list[dict[str, float]]:
#         text = "\n".join(texts)
#         scores = essay_scorer.get_feats(text)
#         print("h")
#         return []


class OpenAIProcessing:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        if api_key is None:
            self.openai = OpenAI(
                base_url=f"http://{url}:{port}/v1/",
                api_key="ollama"
            )
        else:
            self.openai = OpenAI(api_key=api_key)
        self.seed = seed
        self.temperature = temperature
        self.url = url
        self.port = port

    def process_messages(self, model_name, messages):
        return self.openai.chat.completions.create(
            model=model_name,
            seed=self.seed,
            messages=messages,
            temperature=self.temperature,
        ).to_dict()

    def get_llm_runnable(self, model_name: str):
        def _call(inputs):
            prompt = str(inputs)
            messages = [{"role": "user", "content": prompt}]
            result = self.process_messages(model_name=model_name, messages=messages)
            return result["choices"][0]["message"]["content"]

        return RunnableLambda(_call)


if __name__ == '__main__':
    device_i = "cuda" if torch.cuda.is_available() else "cpu"
    # model_test_name = "KevSun/Engessay_grading_ML"

    texts = [
        "It is important for all towns and cities to have large public spaces such as squares and parks. "
        "Do you agree or disagree with this statement? It is crucial for all metropolitan cities and towns to "
        "have some recreational facilities like parks and squares because of their numerous benefits. A number of "
        "arguments surround my opinion, and I will discuss it in upcoming paragraphs. To commence with, the first "
        "and the foremost merit is that it is beneficial for the health of people because in morning time they can "
        "go for walking as well as in the evenings, also older people can spend their free time with their loved ones, "
        "and they can discuss about their daily happenings. In addition, young people do lot of exercise in parks and "
        "gardens to keep their health fit and healthy, otherwise if there is no park they glue with electronic gadgets "
        "like mobile phones and computers and many more. Furthermore, little children get best place to play, they play "
        "with their friends in parks if any garden or square is not available for kids then they use roads and streets "
        "for playing it can lead to serious incidents. Moreover, parks have some educational value too, in schools, "
        "students learn about environment protection in their studies and teachers can take their pupils to parks because "
        "students can see those pictures so lively which they see in their school books and they know about importance "
        "and protection of trees and flowers. In recapitulate, parks holds immense importance regarding education, health "
        "for people of every society, so government should build parks in every city and town."
    ]

    model_test_name = "JacobLinCool/IELTS_essay_scoring_safetensors"
    # scorer = EssayScorer(model_test_name, device=device_i)
    # scores = scorer.process_texts(texts)
    # for i, score in enumerate(scores):
    #     print(f"Essay {i+1} Scores: {score}")
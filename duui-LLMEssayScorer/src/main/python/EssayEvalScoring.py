from run_text import score_essays, load_prompt
import json_repair
from EssayScorer import OpenAIProcessing
import argparse
prompt_template_generation = """
For each given score, generate a complete answer to the following task.

#### Task:
{task}

#### Scoring Rubric:
{scoring_rubric}

Instructions:
1. For each score in the rubric, write an essay matching the quality described for that score.
2. Provide a short explanation justifying why the essay fits the score.
3. Output strictly in the following JSON format (no extra text, markdown, or commentary):

json'''
[
    {{
        "essay": "<Essay for Score 1>",
        "explanation": "<Reason for Score 1>",
        "content": 1
    }},
    {{
        "essay": "<Essay for Score 2>",
        "explanation": "<Reason for Score 2>",
        "content": 2
    }},
    ...
    {{
        "essay": "<Essay for Score N>",
        "explanation": "<Reason for Score N>",
        "content": N
    }}
]
'''
"""

essay_set_descriptions = [
    {
        # "prompt": "Schreibe einen Aufsatz über Umweltschutz.",
        "scoring_rubric": {
            "content": {
                1: {"description": "Unzureichend", "typical_elements": ["Keine klare Idee"],
                    "fine_grained_rubric": ""},
                2: {"description": "In Ordnung", "typical_elements": ["Einige relevante Punkte"],
                    "fine_grained_rubric": ""},
                3: {"description": "Sehr gut", "typical_elements": ["Klare Argumentation"],
                    "fine_grained_rubric": ""}
            }
        },
        "single_evaluator_score_ranges": [[1, 3]],
        # example essays for each score
    }
]

model_prefix="You are a helpful assistant that evaluates essays.",
model_suffix= "Please provide a score based on the content of the essay."

class EssayEvalScoring:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name: str, task: str, essay: str) -> dict:
        scoring_rubric = essay_set_descriptions[0]["scoring_rubric"]
        messages = [
            {"role": "user", "content": prompt_template_generation.format(task=task, scoring_rubric=scoring_rubric)}
        ]
        output_eval = {
            "messages": messages,
            "category": "Essay Evaluation",
            "task": task,
            "essay": essay,
            "model_name": model_name
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_eval_content = result["choices"][0]["message"]["content"]
        output_eval["output"] = output_eval_content
        essay_set_meta = essay_set_descriptions
        try:
            output_eval_json = output_eval_content.split("</think>")[0].strip()
            output_eval_json = json_repair.loads(output_eval_json)
            essay_set_meta[0]["prompt"] = task
            essay_set_meta[0]["examples"] = output_eval_json
        except Exception as e:
            essay_set_meta[0]["prompt"] = task
            essay_set_meta[0]["examples"] = [
                {
                    "essay": "This is a sample essay for score.",
                    "explanation": "The Text explains the reasons.",
                    "content": 3
                }
            ]
        meta_data = essay_set_meta[0]
        parser = argparse.ArgumentParser()
        parser.add_argument("--logging_data_path", type=str, default="./log.log")
        parser.add_argument("--model", type=str, default="mistral", choices=["llama", "mistral"])
        parser.add_argument("--model_size", type=str, default="7b", choices=["7b", "13b"])
        parser.add_argument("--temperature", type=float, default=0.00)
        parser.add_argument("--max_length", type=int, default=4096)
        parser.add_argument("--prompt", type=str, default="holistic_scoring_prompt1")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--dataset_split", type=str, default="test")
        parser.add_argument("--setting", type=str, default="one-shot", choices=["one-shot", "few-shot"])
        parser.add_argument("--full-rubric", action="store_true")
        parser.add_argument("--prompt-template", type=int, default=1, choices=[1, 2, 3, 4])
        parser.add_argument("--instruction-variant", type=int, default=1, choices=[1, 2, 3, 4])
        # args = parser.parse_args()
        args, unknown = parser.parse_known_args()
        extract_scores, instruction, prompt_template = load_prompt(args)
        essay_input = {
            "essay": essay,
            "id": 1,
            "model_prefix": model_prefix,
            "model_suffix": model_suffix
        }
        try:
            essay_score_result = score_essays(
                essays=[essay_input],
                meta_data=meta_data,
                llm=self.openai.get_llm_runnable(model_name=model_name),
                prompt_template=prompt_template,
                instruction=instruction,
                setting="one-shot",
                extract_scores=True
            )
        except Exception as e:
            meta_data["examples"] = [
                {
                    "essay": "This is a sample essay for testing.",
                    "explanation": "The Text explains the reasons.",
                    "content": 3
                }
            ]
            essay_score_result = score_essays(
                essays=[essay_input],
                meta_data=meta_data,
                llm=self.openai.get_llm_runnable(model_name=model_name),
                prompt_template=prompt_template,
                instruction=instruction,
                setting="one-shot",
                extract_scores=True
            )
        output_score_content = essay_score_result[0]["output"]
        output_scoring = {
            "category": "Essay Scoring",
            "task": task,
            "essay": essay,
            "model_name": model_name
        }
        output_scoring["output"] = output_score_content
        output_scoring["result"] = essay_score_result
        try:
            output_score_json = output_score_content.split("</think>")[-1].strip()
            # beginning = output_score_json.find('{')
            # end_index = output_score_json.rfind('}') + 1
            # output_score_json = output_score_json[beginning:end_index]
            output_score_json = json_repair.loads(output_score_json)
            # if isinstance(output_score_json, list):
            #     score = output_score_json[0]["content"]
            # else:
            score = output_score_json["content"]
            if "explanation" in output_score_json:
                explanation = output_score_json["explanation"]
            else:
                explanation = "No explanation provided."
        except Exception as e:
            print(f"Error parsing JSON output: {e}")
            score = -1
            explanation = "Failed to parse JSON output from the model response."
        output_scoring["score"] = score
        output_scoring["reason"] = explanation
        output_eval_scoring = {
            "Generating": output_eval,
            "Scoring": output_scoring,
        }
        return output_eval_scoring


if __name__ == '__main__':
    prompt_i = "Write an essay about environmental protection."
    scoring_rubric_i = essay_set_descriptions[0]["scoring_rubric"]
    model_name = f"test:DeepSeek-R1"
    server_id = "localhost"
    server_port = 11434
    seed = 42
    temperature = 1.0
    api_key = None
    essay_eval_scorer = EssayEvalScoring(url=server_id, port=server_port, seed=seed, temperature=temperature, api_key=api_key)
    result = essay_eval_scorer.run_message(model_name=model_name, task=prompt_i, essay="This is a sample essay for testing.")
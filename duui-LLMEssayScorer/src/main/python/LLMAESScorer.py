#adapted from https://github.com/Xiaochr/LLM-AES/blob/main/inference_slow_module.py
import json
from EssayScorer import OpenAIProcessing
from LLMAESDataset import *


true_rubrics = """
**Content Dimension (8 points in total)**
- Level 1: 6-8 points:
    - Content is complete with appropriate details
    - Expression is closely related to the topic
- Level 2: 3-5 points:
    - Content is mostly complete
    - Expression is fundamentally related to the topic
- Level 3: 0-2 points:
    - Content is incomplete
    - Expression is barely related or completely unrelated to the topic

**Language Dimension (8 points in total)**
- Level 1: 6-8 points:
    - Language is accurate with diverse sentence structures and little or no errors (2 errors or fewer, 8 points; 3-4 errors, 7 points; 5-6 errors, 6 points)
    - Language expression is mostly appropriate
- Level 2: 3-5 points:
    - Language is not quite accurate, with some variation in sentence structures and several errors, but they don't impede understanding (7-8 errors, 5 points; 9-10 errors, 4 points; 11-12 errors, 3 points)
    - Language expression is somewhat inappropriate
- Level 3: 0-2 points:
    - Language is hopelessly inaccurate with numerous language errors, hindering understanding (more than 12 errors)
    - Language expression is completely inappropriate

**Structure Dimension (4 points in total)**
- Level 1: 3-4 points:
    - Clearly and logically structured
    - Smooth and coherent transitions.
- Level 2: 1-2 points:
    - Mostly clearly and logically structured
    - Relatively smooth and coherent transitions
- Level 3: 0-1 point:
    - Not clearly and logically structured
    - Fragmented and disconnected structures and sentences.

**Grading Rules**
- When grading, focus first on the content dimension. If the essay's content is unrelated to the topic, the content score is 0, and both the language and structure scores are also 0. After determining the content score, then focus on the language dimension; generally, the language dimension level will not be higher than the content dimension level. Finally, evaluate the structure score, and then sum the three scores to obtain the overall score.
- For scoring the three dimensions, after initially determining the level, it is generally recommended to first assign the midpoint of the level (for example, if the language dimension is determined to be Level 1, initially assign 7 points). Then, adjust the score by adding or subtracting points to obtain the final specific score.
"""

essay_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an experienced English teacher analyzing high school students' essays according to a specific rubric. Evaluate the following essay based on three dimensions: Content, Language, and Structure, and provide the overall assessment. 

Please provide your evaluation in the following JSON format:
{{
	"content": {{
		"completeness": "Check whether the content is complete, does the essay cover all required points? Provide your explanations with examples in the essay.",
		"topic_relevance": "Does the essay closely related to the given topic? If the content is irrelevant to the topic, then assign score 0.",
		"content_details": "Are the details and expression sufficient? Provide your explanations with examples in the essay.",
		"score_level": "Determine content dimension level based on the analyses above.",
		"score_point": "Determine content score based on the analyses above and the score level."
	}}, 
	"language": {{
		"error_details": "List all the grammar or spelling errors in the essay.",
		"error_cnt": "Total number of errors.",
		"accuracy_and_diversity": "Check the language accuracy, appropriateness, and diversity. Provide your explanations with examples in the essay.",
		"score_level": "Determine language dimension level based on the analyses above.",
		"score_point": "Determine language score based on the analyses above and the score level."
	}}, 
	"structure": {{
		"clarity": "Check whether the structure is clear and logical. Provide your explanations with examples in the essay.",
		"coherence": "Check for smooth and coherent transitions. Provide your explanations with examples in the essay.",
		"score_level": "Determine language dimension level based on the analyses above.",
		"score_point": "Determine language score based on the analyses above and the score level."
	}}, 
	"overall": {{
		"overall_assessment": "The overall assessment of the essay.",
		"score_point": "Determine the overall score."
	}}
}}

### Input:
Scoring rubric:
{}

Essay Prompt:
{}

Student's Essay to Evaluate:
{}

### Response:
{}"""
all_rubrics = [set_1_rubrics, set_2_rubrics, set_3_rubrics, set_4_rubrics, set_5_rubrics, set_6_rubrics, set_7_rubrics, set_8_rubrics]
all_examples = [set_1_examples, set_2_examples, set_3_examples, set_4_examples, set_5_examples, set_6_examples, set_7_examples, set_8_examples]

def zeroshot_norubrics_prompt(essay_prompt, essay):
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Sample Essay Prompt: 
{}

Student's Essay to Evaluate
{}

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]""".format(essay_prompt, essay)
    messages_no_rubric=[
        {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays."},
        {"role": "user", "content": prompt}
    ]
    return messages_no_rubric


def zeroshot_rubrics_prompt(rubrics, essay_prompt, essay):
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

Student's Essay to Evaluate: 
{}

Task Breakdown:
1. Carefully read the provided essay prompt, scoring guidelines, and the student's essay.
2. In the Explanations part, identifying specific elements in the essay referring to the rubrics. In the language dimension, list all the spelling and grammar errors, and count the number of them to determine the Language Score. The Explanations for each dimension should be as detailed as possible.
3. Determine the appropraite scores according to the analysis above. 

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]
""".format(rubrics, essay_prompt, essay)
    messages_zeroshot=[
        {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics."},
        {"role": "user", "content": prompt}
    ]
    return messages_zeroshot

def fewshot_rubrics_prompt(rubrics, essay_prompt, examples, essay):
    prompt = """
As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics and graded examples. You are to act as an impartial judge and evaluate the essays based on the quality of the writing and adherence to the essay prompt.

Three dimensions of scores and a total score should be assigned, where higher scores represent higher quality: 
1. Content (on a scale of 0 to 8)
2. Language (on a scale of 0 to 8)
3. Structure (on a scale of 0 to 4)
4. Total Score (on a scale of 0 to 20) = Content Score + Language Score + Structure Score

Here are the specific guidelines for each score:
{}

Sample Essay Prompt: 
{}

The graded example essays:
{}

Student's Essay to Evaluate: 
{}

Task Breakdown:
1. Carefully read the provided essay prompt, scoring guidelines, and the student's essay.
2. In the Explanations part, identifying specific elements in the essay referring to the rubrics. In the language dimension, list all the spelling and grammar errors, and count the number of them to determine the Language Score. The Explanations for each dimension should be as detailed as possible.
3. Determine the appropraite scores according to the analysis above. 

Please present your evaluation in the following manner:

Explanations: ...
Content Score: ...

Explanations: ...
Language Score: ...

Explanations: ...
Structure Score: ...

Explanations: ...
Total Score: ...

Your final evaluation: 
[Total Score: ..., Content Score: ..., Language Score: ..., Structure Score: ...]
""".format(rubrics, essay_prompt, examples, essay)

    messages_fewshot=[
        {"role": "system", "content": "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays according to a predetermined set of rubrics and graded examples."},
        {"role": "user", "content": prompt}
    ]

    return messages_fewshot

class ScoreSlowStudent:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name: str, essay: str, task: str) -> dict:
        messages = [
            {"role": "user", "content": essay_prompt.format(true_rubrics, task, essay, "")}
        ]
        output_llm_aes = {
            "messages": messages,
            "category": "Slowmodule",
            "task": task,
            "essay": essay,
            "rubrics": true_rubrics,
            "model_name": model_name

        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_slow = result["choices"][0]["message"]["content"]
        output_llm_aes["output"] = output_slow
        output_llm_aes["result"] = result
        try:
            first_index = output_slow.find('{')
            last_index = output_slow.rfind('}')
            json_string = output_slow[first_index:last_index + 1]
            json_object = json.loads(json_string)
            content_info = {
                "Name": "Content-Score",
                "score": json_object["content"]["score_point"],
                "reason": json_object["content"]["completeness"]
            }
            language_info = {
                "Name": "Language-Score",
                "score": json_object["language"]["score_point"],
                "reason": json_object["language"]["accuracy_and_diversity"]
            }
            structure_info = {
                "Name": "Structure-Score",
                "score": json_object["structure"]["score_point"],
                "reason": json_object["structure"]["clarity"]
            }
            overall_info = {
                "Name": "Overall-Score",
                "score": json_object["overall"]["score_point"],
                "reason": json_object["overall"]["overall_assessment"]
            }
            output_llm_aes["content_info"] = content_info
            output_llm_aes["language_info"] = language_info
            output_llm_aes["structure_info"] = structure_info
            output_llm_aes["overall_info"] = overall_info
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            output_llm_aes["error"] = "Failed to parse JSON from response"
        return output_llm_aes

class ScoreStudent:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name, category, set_prompt, set_essay, set_examples: str = "", set_rubrics: str = ""):
        if category == "zeroshot_norubrics":
            message_input = zeroshot_norubrics_prompt(set_prompt, set_essay)
            output_llm_aes = {
                "message": message_input,
                "category": "zeroshot_norubrics",
                "task": set_prompt,
                "essay": set_essay,
                "rubrics": "",
                "examples": ""
            }
        elif category == "zeroshot":
            message_input = zeroshot_rubrics_prompt(set_rubrics, set_prompt, set_essay)
            output_llm_aes = {
                "message": message_input,
                "category": "zeroshot",
                "task": set_prompt,
                "essay": set_essay,
                "rubrics": set_rubrics,
                "examples": ""
            }

        else:
            raise ValueError("Invalid category")
        result = self.openai.process_messages(model_name=model_name, messages=message_input)
        output_result = result["choices"][0]["message"]["content"]
        output_llm_aes["output"] = output_result
        output_llm_aes["model_name"] = model_name
        output_llm_aes["result"] = output_result
        # Extract Information
        try:
            # content Score with Explanations
            split_exp = output_result.split("Explanations:")
            content_score = split_exp[1].split("Content Score:")[1].strip().split("/")[0].strip()
            # remove all non-numeric characters from content_score
            content_score = ''.join(filter(str.isdigit, content_score))
            content_explanation = split_exp[1].split("Content Score:")[0].strip()
            output_llm_aes["content_score"] = {
                "score": float(content_score),
                "reason": content_explanation,
                "name": "ContentScore"
            }
            # language Score with Explanations
            language_score = output_result.split("Language Score:")[1].strip().split("/")[0].strip()
            # remove all non-numeric characters from language_score
            language_score = ''.join(filter(str.isdigit, language_score))
            language_explanation = split_exp[2].split("Language Score:")[0].strip()
            output_llm_aes["language_score"] = {
                "score": float(language_score),
                "reason": language_explanation,
                "name": "LanguageScore"
            }
            # structure Score with Explanations
            structure_score = output_result.split("Structure Score:")[1].strip().split("/")[0].strip()
            # remove all non-numeric characters from structure_score
            structure_score = ''.join(filter(str.isdigit, structure_score))
            structure_explanation = split_exp[3].split("Structure Score:")[0].strip()
            output_llm_aes["structure_score"] = {
                "score": float(structure_score),
                "reason": structure_explanation,
                "name": "StructureScore"
            }
            # total Score with Explanations
            total_score = output_result.split("Total Score:")[1].strip().split("/")[0].strip()
            # remove all non-numeric characters from total_score
            total_score = ''.join(filter(str.isdigit, total_score))
            # total_explanation = split_exp[4].split("Total Score:")[0].strip()
            output_llm_aes["total_score"] = {
                "score": float(total_score),
                # "reason: total_explanation,
                "name": "TotalScore"
            }
        except Exception as e:
            print(f"Error processing the output: {e}")
        #norubrics
        try:
            content_score = output_result.split("**Content ")[1].strip().split("/")[0].strip().split("\n")[0].strip()
            # remove all non-numeric characters from content_score
            content_score = ''.join(filter(str.isdigit, content_score))
            content_explanation = output_result.split("**Content ")[1].strip().split("\n")[1].strip()
            output_llm_aes["content_score"] = {
                "score": float(content_score),
                "reason": content_explanation,
                "name": "ContentScore"
            }
            language_score = output_result.split("**Language ")[1].strip().split("/")[0].strip().split("\n")[0].strip()
            # remove all non-numeric characters from language_score
            language_score = ''.join(filter(str.isdigit, language_score))
            language_explanation = output_result.split("**Language ")[1].strip().split("\n")[1].strip()
            output_llm_aes["language_score"] = {
                "score": float(language_score),
                "reason": language_explanation,
                "name": "LanguageScore"
            }
            structure_score = output_result.split("**Structure ")[1].strip().split("/")[0].strip().split("\n")[0].strip()
            # remove all non-numeric characters from structure_score
            structure_score = ''.join(filter(str.isdigit, structure_score))
            structure_explanation = output_result.split("**Structure ")[1].strip().split("\n")[1].strip()
            output_llm_aes["structure_score"] = {
                "score": float(structure_score),
                "reason": structure_explanation,
                "name": "StructureScore"
            }
            total_score = output_result.split("**Total ")[1].strip().split("/")[0].strip().split("\n")[0].strip()
            # remove all non-numeric characters from total_score
            total_score = ''.join(filter(str.isdigit, total_score))
            output_llm_aes["total_score"] = {
                "score": float(total_score),
                # "reason: total_explanation,
                "name": "TotalScore"
            }
        except Exception as e:
            print(f"Error processing the output without rubrics: {e}")
        return output_llm_aes
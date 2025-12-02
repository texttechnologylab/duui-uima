import json
import os
import re
from typing import List, Dict, Union

import torch
from datasets.utils.logging import disable_progress_bar
# from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
# from langchain.llms import VLLM
from langchain_community.llms import VLLM
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from tqdm import tqdm

from essay_meta_data import essay_set_descriptions

disable_progress_bar()


def load_model(config, prompt_template):
    """
        Load a specified LLM model and configure its prompt template with model-specific prefixes and suffixes.

        Parameters:
        - config (object): Configuration object containing model specifications such as type, size, maximum length, and temperature.
        - prompt_template (PromptTemplate): A template object for generating prompts to the LLM.

        Returns:
        - tuple: A tuple containing the loaded LLM model and the updated prompt template.
    """
    # load llm
    if config.model == "llama":
        llm = load_llama2_vllm(max_length=config.max_length, temperature=config.temperature,
                               model_size=config.model_size)
    elif config.model == "mistral":
        llm = load_mistral_vllm(max_length=config.max_length, temperature=config.temperature)
    else:
        raise ValueError("Model not supported")
    prompt_template = prompt_template.partial(model_prefix="<s>[INST]", model_suffix="[/INST]")
    return llm, prompt_template


def collect_llm_output(llm, inputs, output_parser, prompt_template, extract_scores: bool = True):
    """
    Collect outputs from the LLM for a given set of inputs, optionally parsing the outputs to extract scores.

    Parameters:
    - llm (VLLM): The loaded LLM model to use for generating outputs.
    - inputs (list or dict): The input data to be processed by the LLM.
    - output_parser (StructuredOutputParser): The parser to use for extracting structured information from LLM outputs.
    - prompt_template (PromptTemplate): The template object for generating prompts to the LLM.
    - extract_scores (bool, optional): Flag to determine whether to parse outputs to extract scores. Defaults to True.

    Returns:
    - list or dict: The LLM outputs, possibly enriched with parsed data if extract_scores is True.
    """
    if extract_scores:
        # get the format instructions from the output parser and use only the json markdown example from it
        format_instructions = output_parser.get_format_instructions().split("\n\n")[-1]
        prompt_template = prompt_template.partial(format_instructions=format_instructions)

    chain = prompt_template | llm
    print('------')
    # print('prompt_template:', prompt_template)
    # print('llm:', llm)
    # print('chain:', chain)
    if isinstance(inputs, dict):
        inputs = [inputs]
    print('before chain.batch')
    # print('type(chain):', type(chain))
    # print('type(inputs):', type(inputs))
    # print('inputs[0].keys():', inputs[0].keys())
    # # print('inputs[0].items()[0]:', list(inputs[0].items())[0])
    # print('inputs[0][essay]', inputs[0]['essay'])
    print('inputs[0]', inputs[0])
    # print('inputs[0][id]', inputs[0]['id'])
    # print('inputs[1][essay]', inputs[1]['essay'])
    # print('inputs[1][id]', inputs[1]['id'])
    llm_outputs = chain.batch(inputs)
    print('llm_outputs:', llm_outputs)
    if not extract_scores:
        for i, output in enumerate(llm_outputs):
            inputs[i].update({"output": output})
        return inputs

    return parse_outputs(llm_outputs, inputs, llm=llm, output_parser=output_parser)


def parse_outputs(outputs, inputs, output_parser, llm):
    """
    Parse the outputs from the LLM to extract structured data, using both the output parser and regex methods.

    Parameters:
    - outputs (list): The raw outputs from the LLM.
    - inputs (list): The corresponding inputs for which the outputs were generated.
    - output_parser (StructuredOutputParser): The parser to use for extracting structured information from LLM outputs.
    - llm (VLLM): The LLM model used for generating outputs.

    Returns:
    - list: A list of dictionaries, each containing the original input, the raw LLM output, and the parsed output.
    """
    parser_template = """<s> [INST] 
    This feedback was written: 
    '''
    {feedback}
    '''
    Extract the score for this essay in this form: {format_instructions}. 
    If there is anything that is not provided in the feedback, just write "None" instead.
    [/INST] """
    parser_prompt = PromptTemplate(template=parser_template, input_variables=["format_instructions", "feedback"])

    llm_tasks = []
    results = []

    # try to use the simple methods first and collect the unparsable ones for a batch input to the llm
    for output, current_item in zip(outputs, inputs):
        parsed_output = try_output_parser_and_regex(output, output_parser)
        if parsed_output is None:
            llm_tasks.append((output, current_item))
        else:
            current_item.update({
                "output_parsing_info": {"method": "output_parser", "llm": None, "llm_output": None,
                                        "output": parsed_output,
                                        "format_instructions": None},
                "output": output,
                "parsed_output": parsed_output,
            })
            results.append(current_item)

    # prompt the llm again to convert the unparsable outputs into a valid JSON format
    if len(llm_tasks) > 0:
        format_instructions = output_parser.get_format_instructions()
        formatted_parser_prompt = parser_prompt.partial(format_instructions=format_instructions)
        chain = formatted_parser_prompt | llm
        llm_outputs = chain.batch([{"feedback": output} for output, _ in llm_tasks])
        for i, output in enumerate(llm_outputs):
            feedback, current_item = llm_tasks[i]
            parsed_output = try_output_parser_and_regex(output, output_parser)
            current_item.update({
                "output_parsing_info": {"method": "llm", "llm": str(llm), "parsing_llm_output": output,
                                        "output": parsed_output,
                                        "format_instructions": format_instructions},
                "output": feedback,
                "parsed_output": parsed_output,
            })
            results.append(current_item)

    return results


def try_output_parser_and_regex(output, output_parser):
    """
    Attempt to parse LLM output using structured output parser and regex methods to extract structured data.

    Parameters:
    - output (str): The raw output from the LLM.
    - output_parser (StructuredOutputParser): The parser to use for extracting structured information from LLM outputs.

    Returns:
    - dict or None: The parsed output as a dictionary, or None if parsing fails.
    """
    try:
        result = parse_output_with_json_regex(output)[0]
        result = remove_comments_in_json(result)
        result = remove_highscores_json(result)
        result = remove_final_comma_in_json(result)
        return output_parser.parse(
            f"```json\n{result}\n```")  # insert the json code block because the regex is not capturing it

    except Exception:
        return None


def remove_final_comma_in_json(string):
    """
    Remove the final comma in a JSON string, but only for the last row.

    Parameters:
    - string (str): The JSON string to process.

    Returns:
    - str: The processed JSON string with the final comma removed.
    """
    pattern = r',\s*}\s*$'
    return re.sub(pattern, "\n}", string)


def remove_highscores_json(string):
    """
    Remove the highscore fractions from a JSON string.

    Parameters:
    - string (str): The JSON string to process.

    Returns:
    - str: The processed JSON string with highscore fractions removed.
    """
    lines = string.split("\n")
    # try to match the line to "key": value_without_quotes pattern
    pattern = r'(\s*".*?":\s*?)(\d+/\d+)(,?)'
    for i in range(len(lines)):
        match = re.match(pattern, lines[i])
        if match is not None:
            lines[i] = match.group(1) + match.group(2).split("/")[0] + match.group(3)
    return "\n".join(lines)


def remove_comments_in_json(string):
    """
    Remove comments from a JSON string.

    Parameters:
    - string (str): The JSON string to process.

    Returns:
    - str: The processed JSON string with comments removed.
    """
    lines = string.split("\n")
    # try to match the line to "key": value_without_quotes pattern
    pattern = r'(\s*".*?":\s*?)(.*)(\s*?)(//.*)'
    for i in range(len(lines)):
        match = re.match(pattern, lines[i])
        if match is not None:
            lines[i] = match.group(1) + match.group(2) + match.group(3)
    return "\n".join(lines)


def parse_output_with_json_regex(output):
    """
    Extract JSON strings from LLM output using regex.

    Parameters:
    - output (str): The raw output from the LLM.

    Returns:
    - list: A list of extracted JSON strings.
    """
    pattern = r'({(?:[^{}]|\{[^{}]*\})*})\s*'
    return re.findall(pattern, output, re.DOTALL)


def format_instruction(instruction, meta_data, setting):
    """
    Format the instruction for LLM processing by replacing placeholders with actual values based on essay metadata and
    setting.

    Parameters:
    - instruction (str): The base instruction with placeholders.
    - meta_data (dict): The metadata related to the essay set.
    - setting (str): The setting for instruction formatting, e.g., "one-shot" or "few-shot".

    Returns:
    - str: The formatted instruction with placeholders replaced by actual values.
    """
    scoring_instructions = "\n".join([
        f"{score_type}: from {meta_data['single_evaluator_score_ranges'][j][0]} to {meta_data['single_evaluator_score_ranges'][j][1]}"
        for j, score_type in enumerate(meta_data["scoring_rubric"])])

    # Generate rubrics and examples
    rubric_small, rubric_full = generate_rubics(meta_data)
    examples = generate_examples(meta_data, setting=setting)

    # Create a dictionary with all possible values
    all_values = {
        'scoring_instructions': scoring_instructions,
        'rubric_small': rubric_small,
        'rubric_full': rubric_full,
        'examples': examples
    }

    # Find all placeholders (marked by {}) in the instruction
    placeholders = re.findall(r'\{(.*?)\}', instruction)

    # Filter the values dictionary to only include keys that are placeholders in the instruction
    values_to_use = {key: all_values[key] for key in placeholders if key in all_values}

    # Fill the instruction using format_map
    return instruction.format_map(values_to_use)


def generate_rubics(meta_data):
    """
    Generates small and full rubrics based on essay set metadata.

    Parameters:
    - meta_data (dict): The metadata for the essay set, including scoring rubric details.

    Returns:
    - tuple: A tuple containing the small rubric and the full rubric as strings.
    """
    rubric_small = ""
    rubric_full = ""
    for criteria in meta_data["scoring_rubric"]:
        rubric_small += f"Scoring rubric for '{criteria}':\n"
        rubric_full += f"Scoring rubric for '{criteria}':\n"
        for points in meta_data["scoring_rubric"][criteria]:
            description = meta_data["scoring_rubric"][criteria][points]['description']
            rubric_small += f"{points} points: {description}\n"
            typical_elements = "\n- ".join(meta_data["scoring_rubric"][criteria][points]["typical_elements"])
            fine_grained_rubric = meta_data["scoring_rubric"][criteria][points]["fine_grained_rubric"]
            rubric_full += f"{points} points: {description}\n"
            if len(typical_elements) > 0:
                rubric_full += f"Typical elements: \n- {typical_elements}\n\n"
            if len(fine_grained_rubric) > 0:
                rubric_full += f"Fine-grained rubric: {fine_grained_rubric}"
            rubric_full += "\n\n"

    return rubric_small, rubric_full


def generate_examples(meta_data, setting="one-shot", max_length=1024 * 5):
    """
    Generates example essays and their evaluations based on the essay set metadata and specified setting.

    Parameters:
    - meta_data (dict): The metadata for the essay set, including example essays.
    - setting (str, optional): The instruction setting, either "one-shot" or "few-shot". Defaults to "one-shot".
    - max_length (int, optional): The maximum length of the generated examples string. Defaults to 5120 characters.

    Returns:
    - str: Formatted examples including essays, their evaluations, and scores.
    """
    assert setting in ["one-shot", "few-shot"], "Setting must be either 'one-shot' or 'few-shot'"
    examples = meta_data["examples"]
    if setting == "one-shot":
        examples = examples[:1]

    formatted_examples = ""
    for i, example in enumerate(examples):
        scores = {score_type: example[score_type] for score_type in meta_data["scoring_rubric"]}
        added_text = f"Example {i}:\n Essay: `{example['essay']}`\n Reasoning: {example['explanation']}\n Scores: ```json\n{json.dumps(scores, indent=4)}\n```\n"

        # check if the added text is too long and stop adding examples if it is
        if len(formatted_examples) + len(added_text) > max_length:
            break
        else:
            formatted_examples += added_text

    return formatted_examples


NUM_ESSAY_SETS = 8


def analyse_dataset(llm, dataset, instruction, prompt_template, setting, logging_data_path=None,
                    split="test", extract_scores=True):
    """
    Analyze an essay dataset by scoring each essay across all folds and essay sets, logging the results.

    Parameters:
    - llm (VLLM): The LLM model to use for scoring essays.
    - dataset (dict): The dataset containing essays, organized by folds and splits.
    - instruction (str): The instruction to guide the LLM in scoring essays.
    - prompt_template (PromptTemplate): The template for generating prompts to the LLM.
    - setting (str): The setting for the analysis, affecting instruction formatting.
    - response_schemas (list, optional): List of ResponseSchema objects for structuring LLM outputs. Defaults to an empty list.
    - logging_data_path (str, optional): Path to save logging data. If None, logging is not performed. Defaults to None.
    - split (str, optional): The dataset split to analyze (e.g., "test", "val"). Defaults to "test".
    - extract_scores (bool, optional): Flag indicating whether to extract scores from LLM outputs. Defaults to True.

    Returns:
    - None
    """
    # load the logging data if it exists to not make the same computations again
    logging_data = {}
    if logging_data_path is not None and os.path.exists(logging_data_path):
        with open(logging_data_path, "r") as f:
            logging_data = json.load(f)

    # rename the split if needed
    if split in ["val", "validation"]:
        split = "dev"

    # create a list of tuples for better tqdm progress bar
    fold_essay_sets = [(fold, i) for fold in list(dataset.keys()) for i in range(NUM_ESSAY_SETS)]

    # iterate over all the folds and compute the qwk for each essay set for each fold
    # we need to treat each essay set differently, as the scoring range as well as the prompt
    # is different
    for fold, i in tqdm(fold_essay_sets):
        if f"{fold}_{i}" in logging_data:
            print(f"Skipping fold {fold} and essay set {i} as it is already logged")
            continue

        # create the sub-dataset that contains all the essays for the current essay set in this fold
        essays = [{"essay": essay["essay"], "id": int(essay["essay_id"])} for essay in
                  dataset[fold][split].filter(lambda row: row["essay_set"] == i + 1)]

        meta_data = essay_set_descriptions[i]

        outputs = score_essays(essays, meta_data, llm, prompt_template, instruction, setting,
                               extract_scores=extract_scores)

        logging_data[f"{fold}_{i}"] = outputs
        # save the logging data as a file
        with open(logging_data_path, "w") as f:
            json.dump(logging_data, f)


def score_essays(essays: List[Dict[str, Union[str, int]]], meta_data: Dict, llm, prompt_template: PromptTemplate,
                 instruction: str, setting, extract_scores: bool = True) -> List[Dict[str, Union[str, int]]]:
    """
    Score a list of essays using an LLM, optionally extracting scores from the LLM's outputs.

    Parameters:
    - essays (list): A list of dictionaries, each representing an essay with keys "essay" and "id".
    - meta_data (dict): Metadata for the essay set being scored, including scoring rubric and examples.
    - llm (VLLM): The LLM model to use for scoring essays.
    - prompt_template (PromptTemplate): The template for generating prompts to the LLM.
    - instruction (str): The instruction to guide the LLM in scoring essays.
    - setting (str): The setting for scoring, affecting instruction formatting.
    - extract_scores (bool, optional): Flag indicating whether to extract scores from LLM outputs. Defaults to True.

    Returns:
    - list: A list of dictionaries, each containing an essay id, the original essay, the raw LLM output, and optionally the parsed output.
    """
    formatted_instruction = format_instruction(instruction, meta_data, setting=setting)
    prompt = prompt_template.partial(instruction=formatted_instruction, prompt=meta_data["prompt"])

    if extract_scores:
        # Configure the output parser that is using JSON to parse the output
        response_schemas = [
            ResponseSchema(name=score_type, description=f"the {score_type} score for the essay", type="int") for
            score_type in meta_data["scoring_rubric"]]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    else:
        output_parser = None

    # decompose all essays into batches and send them to the llm
    outputs = collect_llm_output(llm, essays, prompt_template=prompt, output_parser=output_parser,
                                 extract_scores=extract_scores)
    return outputs


# ============ LOADING LLM MODELS =========================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def load_llama2_vllm(max_length=4096, temperature=0.01, model_size="7b"):
    """
    Load a LLAMA-2 model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.
    - model_size (str, optional): The size of the model (e.g., "7b"). Defaults to "7b".

    Returns:
    - VLLM: The loaded LLAMA-2 model.
    """
    return VLLM(model=f"meta-llama/Llama-2-{model_size}-chat-hf", max_length=max_length, temperature=temperature)


def load_mistral_vllm(max_length=4096, temperature=0.01):
    """
    Load a Mistral model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    return VLLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_length=max_length, temperature=temperature)


def load_gpt4_vllm(max_length=4096, temperature=0.01):
    """
    Load a Mistral model with specified parameters.

    Parameters:
    - max_length (int, optional): The maximum token length for the model's outputs. Defaults to 4096.
    - temperature (float, optional): The temperature for sampling outputs. Defaults to 0.01.

    Returns:
    - VLLM: The loaded Mistral model.
    """
    return VLLM(model="mistralai/Mistral-7B-Instruct-v0.2", max_length=max_length, temperature=temperature)
import pandas as pd
from EssayScorer import OpenAIProcessing

def Vanilla_compose_prompt(msg_system, msg_user_instruction):
    messages = [
        {'role': 'user', 'content': msg_system + '\n' + msg_user_instruction}
    ]
    return messages

def MTS_compose_prompt_retrieval(msg_system, msg_user_retrieval):
    messages = [
        {'role': 'user', 'content': msg_system + '\n' + msg_user_retrieval},
    ]
    return messages

def MTS_compose_prompt_score(msg_system, msg_user_retrieval, msg_assistant_retrieval, msg_user_score):
    messages = [
        {'role': 'user', 'content': msg_system + '\n' + msg_user_retrieval},
        {'role': 'assistant', 'content': msg_assistant_retrieval},
        {'role': 'user', 'content': msg_user_score},
    ]
    return messages


class Vanilla_OpenAI:
    def __init__(self, template_path, category, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai  = OpenAIProcessing(url, port, seed, temperature, api_key)
        self.template_path = template_path
        self.category = category
        self.template = self.load_template()

    def load_template(self):
        template = {}
        df = pd.read_excel(self.template_path)
        for index, row in df.iterrows():
            prompt_id = row['prompt_id']
            msg_system = row['msg_system']
            msg_user_instruction_template = row['msg_user_instruction_template']
            template[prompt_id] = {
                'msg_system': msg_system,
                'msg_user_instruction': msg_user_instruction_template,
                'type': 'Vanilla',
                'Category': self.category
            }
        return template

    def run_message(self, template_id, prompt, essay, model_name):
        if template_id not in self.template:
            raise ValueError(f"Template ID {template_id} not found in the template.")

        msg_system = self.template[template_id]['msg_system']
        msg_user_instruction = self.template[template_id]['msg_user_instruction'].replace("@prompt", prompt)
        msg_user_instruction = msg_user_instruction.replace("@essay", essay)

        messages = Vanilla_compose_prompt(msg_system, msg_user_instruction)

        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output = result["choices"][0]["message"]["content"]

        output_score = output.split("</think>")[-1].strip()
        try:
            score_text = output_score.split("Score:")[-1].strip()
            #remove any non-numeric characters
            score_text = ''.join(filter(str.isdigit, score_text))
            score = float(score_text)
        except Exception as e:
            print(f"Error parsing score output: {e}")
            score = -1


        output_i = {
            "score": {
                "result": result,
                "output": output,
                "message": messages,
                "prompt": prompt,
                "essay": essay,
                "trait": "Vanilla",
                "score": score,
                "output_score": output_score,
            }
        }

        return output_i


class MTS_OpenAI:
    def __init__(self, template_path, category, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)
        self.template_path = template_path
        self.category = category
        self.template = self.load_template()

    def load_template(self):
        template = {}
        df = pd.read_excel(self.template_path)
        for index, row in df.iterrows():
            prompt_id = row['prompt_id']
            msg_system = row['msg_system_template']
            msg_user_retrieval_template = row['msg_user_retrieval_template']
            msg_user_score_template = row['msg_user_score_template']
            trait1 = row['trait_1']
            trait2 = row['trait_2']
            trait3 = row['trait_3']
            trait4 = row['trait_4']
            description1 = row['description_1']
            description2 = row['description_2']
            description3 = row['description_3']
            description4 = row['description_4']
            rubric1 = row['rubric_1']
            rubric2 = row['rubric_2']
            rubric3 = row['rubric_3']
            rubric4 = row['rubric_4']
            template[prompt_id] = {
                'msg_system': msg_system,
                'msg_user_retrieval': msg_user_retrieval_template,
                'msg_user_score': msg_user_score_template,
                'trait': {
                    1: trait1,
                    2: trait2,
                    3: trait3,
                    4: trait4
                },
                'description': {
                    1: description1,
                    2: description2,
                    3: description3,
                    4: description4
                },
                'rubric': {
                    1: rubric1,
                    2: rubric2,
                    3: rubric3,
                    4: rubric4
                },
                'type': 'MTS',
                'Category': self.category
            }
        return template

    def run_message(self, template_id, traid_id, task, essay, model_name):
        if template_id not in self.template:
            raise ValueError(f"Template ID {template_id} not found in the template.")
        trait = self.template[template_id]['trait'][traid_id]
        description = self.template[template_id]['description'][traid_id]
        rubric = self.template[template_id]['rubric'][traid_id]
        msg_system = self.template[template_id]['msg_system']
        msg_user_retrieval_template = self.template[template_id]['msg_user_retrieval']
        msg_user_retrieval_template = msg_user_retrieval_template.replace("@prompt", task)
        msg_user_retrieval_template = msg_user_retrieval_template.replace("@essay", essay)
        msg_user_retrieval_template = msg_user_retrieval_template.replace("@trait", trait)
        msg_user_retrieval_template = msg_user_retrieval_template.replace("@description", description)
        msg_user_retrieval_template = msg_user_retrieval_template.replace("@rubric", rubric)

        message_retrieval = MTS_compose_prompt_retrieval(msg_system, msg_user_retrieval_template)
        result_retrieval = self.openai.process_messages(model_name=model_name, messages=message_retrieval)
        output_retrieval = result_retrieval["choices"][0]["message"]["content"]
        output_retrieval = output_retrieval.split("</think>")[-1].strip()

        msg_user_score_template = self.template[template_id]['msg_user_score']
        msg_user_score_template = msg_user_score_template.replace("@prompt", task)
        msg_user_score_template = msg_user_score_template.replace("@essay", essay)
        msg_user_score_template = msg_user_score_template.replace("@trait", trait)
        msg_user_score_template = msg_user_score_template.replace("@description", description)
        msg_user_score_template = msg_user_score_template.replace("@rubric", rubric)

        message_score = MTS_compose_prompt_score(
            msg_system,
            msg_user_retrieval_template,
            output_retrieval,
            msg_user_score_template
        )

        result_score = self.openai.process_messages(model_name=model_name, messages=message_score)
        output_score = result_score["choices"][0]["message"]["content"]
        output_score = output_score.split("</think>")[-1].strip()

        try:
            score_text = output_score.split("Score:")[-1].strip()
            #remove any non-numeric characters
            score_text = ''.join(filter(str.isdigit, score_text))
            score = float(score_text)
        except Exception as e:
            print(f"Error parsing score output: {e}")
            score = -1

        #
        #
        # messages = MTS_compose_prompt_retrieval(msg_system, msg_user_retrieval)

        output_all = {
            "retrieval": {
                "result": result_retrieval,
                "output": output_retrieval,
                "message": message_retrieval,
                "prompt": task,
                "essay": essay,
                "trait": trait,
                "description": description,
                "rubric": rubric,
                "trait_id": traid_id,
            },
            "score": {
                "result": result_score,
                "output": output_score,
                "message": message_score,
                "prompt": task,
                "essay": essay,
                "trait": trait,
                "description": description,
                "rubric": rubric,
                "trait_id": traid_id,
                "score": score,
            }
        }

        return output_all
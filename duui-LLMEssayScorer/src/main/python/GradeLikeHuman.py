from EssayScorer import OpenAIProcessing
import json
import json_repair

# prompt_template = "You are a professional teacher of Operating System course. For evaluting students' tutorial, you are creating grading rubric. You will be given a question, a initial grading rubric, and several examples of correctly grading to some students' answers.\n The current question is: {question}. The initial grading rubric is: {init_rubric}. and the full_points is {full_points}. The samples of correctly grading are:{sample_studata}\n Your generated grading criteria should illustrate the key points of the answer for each question or sub-question and the rules for assigning points.\n Let's think step by step. Once you made a final analyisis result, output the generated rubric in the format: {format_instructions}."

self_reflection = "You already know your [Previous_evalutaion]. please revisit and reflect on your previous assessment. Review the results and consider the insights you've gained. Based on this reflective process, provide an improved and final evaluation results"

prompt_template = """You are a professional teacher of Operating System course. You are evaluating student's tutorial. You will be given a question, a grading rubric, and a student response. Your task in to grade the student's response with a grading score in the range from 0 to {full_points} and give the confidence level of the score.\nThe current question is: {question}. The grading rubric is:{rubric}.\nHere is an example:{example}\nThe student’s response is {studata}. Please compare the student response with grading rubric. Let's think step by step. Once you made a final analysis result, output the evaluation result in the format: {format_instructions}."""

class GradingLikeHumanScorer:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self,model_name: str, question: str, rubric: str, example:str, student_response: str, full_points: int = 10):
        output_score = self.run_message_score(model_name, question, rubric, example, student_response, full_points)
        messages_score = output_score["messages"]
        score_content = output_score["output"].split("</think>")[-1].strip()
        messages_score.append({"role": "assistant", "content": score_content})
        output_reflection = self.run_self_reflection(model_name, messages_score)
        message1 = output_score["output"].split("</think>")[-1].strip()
        message2 = output_reflection["output"].split("</think>")[-1].strip()
        output_extraction = self.run_extraction_score(model_name, message1, message2)
        try:
            output_extraction_content = output_extraction["output"].split("</think>")[-1].strip()
            output_extraction_content = json_repair.loads(output_extraction_content)
            scoring_score = output_extraction_content["Message1"]["score"]
            scoring_confidence = output_extraction_content["Message1"]["confidence"]
            reflection_score = output_extraction_content["Message2"]["score"]
            reflection_confidence = output_extraction_content["Message2"]["confidence"]
            output_extraction["Scoring"] =  {"score": scoring_score, "confidence": scoring_confidence}
            output_extraction["Reflection"] =  {"score": reflection_score, "confidence": reflection_confidence}
        except Exception as e:
            print(f"Error: {e}")
            output_extraction["Scoring"] = {"score": -1, "confidence": "N/A"}
            output_extraction["Reflection"] = {"score": -1, "confidence": "N/A"}
        output_messages = {
            "Scoring": output_score,
            "Reflection": output_reflection,
            "Extraction": output_extraction
        }
        return output_messages


    def run_message_score(self, model_name: str, question: str, rubric: str, example:str, student_response: str, full_points: int = 10) -> dict:
        messages = [
            {"role": "user", "content": prompt_template.format(
                question=question,
                rubric=rubric,
                studata=student_response,
                full_points=full_points,
                example=example,
                format_instructions="Evaluation Result: {score} with confidence level {confidence}"
            )}
        ]
        output_llm_grading = {
            "messages": messages,
            "category": "EssayGrading",
            "task": question,
            "student_answer": student_response,
            "rubric": rubric,
            "model_name": model_name
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_grading = result["choices"][0]["message"]["content"]
        output_llm_grading["output"] = output_grading
        output_llm_grading["result"] = result
        return output_llm_grading

    def run_self_reflection(self, model_name, messages):
        messages.append({"role": "user", "content": self_reflection})
        output_llm_reflection = {
            "messages": messages,
            "model_name": model_name,
            "category": "SelfReflection",
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_reflection = result["choices"][0]["message"]["content"]
        output_llm_reflection["output"] = output_reflection
        output_llm_reflection["result"] = result
        return output_llm_reflection

    def run_extraction_score(self, model_name, message1, message2):
        messages = []
        instruction_template = "The format is\n\t{\"Message1\":\n\t\t{\"score\": <score>,\n\t\t\"confidence\": <confidence>\n\t\t},\n\t\"Message2\":\n\t\t{\"score\": <score>,\n\t\t\"confidence\": <confidence>\n\t\t}\n}"
        template = "Extract the score and confidence level from the [Message1] and [Message2] as Json.\n{instruction}\nGive only the Json output without any additional output.\n\nMessage1: {message1}\n\nMessage2: {message2}"
        messages.append({
            "role": "user",
            "content": template.format(instruction=instruction_template,message1=message1, message2=message2)
        })
        output_llm_extraction = {
            "messages": messages,
            "model_name": model_name,
            "category": "ScoreExtraction",
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_extraction = result["choices"][0]["message"]["content"]
        output_llm_extraction["output"] = output_extraction
        output_llm_extraction["result"] = result
        return output_llm_extraction


if __name__ == '__main__':
    prompt_i = "What is the difference between a process and a thread in operating systems?"
    rubric_i = "A process is an independent program in execution, while a thread is a smaller unit of a process that can run concurrently with other threads within the same process. Processes have their own memory space, while threads share the same memory space of their parent process."
    example_i = "Student i: A process is a program that is running on a computer, while a thread is a smaller part of that program that can run at the same time as other parts. For example, in a web browser, one thread might be loading a webpage while another thread is playing a video.\n Score: 8 with confidence level 90%\n\nStudent ii: A process is like a complete program that runs on its own, while a thread is like a part of that program that can do things at the same time as other parts. For instance, in a game, one thread might handle graphics while another handles user input.\n Score: 10 with confidence level 95%\n\nStudent iii: A process is a program that runs on a computer, and a thread is a part of that program that can run at the same time as other parts. For example, in a web browser, one thread might be loading a webpage while another thread is playing a video.\n Score: 7 with confidence level 85%\n"
    student_response_i = "A process is like a complete program that runs on its own, while a thread is like a part of that program that can do things at the same time as other parts. For instance, in a game, one thread might handle graphics while another handles user input."
    # bad_student_response_i = "Nothing is different between a process and a thread, they are the same thing. A process is just a thread that can run on its own."
    full_points_i = 10
    model_name_test = "test:DeepSeek-R1"
    # link = "anduin.hucompute.org"
    link = "localhost"  # Replace with your server address
    port_i = 11434
    seed_i = 42
    temperature_i = 1.0
    scorer = GradingLikeHumanScorer(link, port_i, seed_i, temperature_i)
    output = scorer.run_message(model_name=model_name_test, question=prompt_i, rubric=rubric_i, example=example_i, student_response=student_response_i, full_points=full_points_i)
    print(output)
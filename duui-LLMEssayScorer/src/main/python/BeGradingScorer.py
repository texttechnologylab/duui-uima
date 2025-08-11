from EssayScorer import OpenAIProcessing
import json

# code_evaluating = {
#     "prompt_generation": """
#     You are an expert programming instructor responsible for evaluating student programming assignments. Write five separate codes that reflect grades from 0 to 4 out of 5 for the solution to the question: {question}. Use the provided model answer as a reference: {Solution}.
#     """,
#     "prompt_evaluating": """
#     You are an expert programming instructor who evaluate programming assignments for students , i want from you to evaluate the code from students in this file , where is column {questionsummary} is the question and column {responsesummary} is the solution of the student.
#     Instruction:
#     - Please the answer response grade out of 5 (0,1,2,3,4,5)
#     - Check The code is right according to the question
#     - Don’t attention on the optimize and the clean code rubric.
#     """,
#     "system_prompt": """
#     As a programming instructor , you ’ve been tasked with assessing student assignments. In the provided file, each entry contains a question (in the column {question}) and a corresponding solution attempted by a student (in the column {response}). Your job is to evaluate how accurately the code addresses the question. Assign scores from 0 to 5, considering the code ’s alignment with the question ’s requirements. Provide scores reflecting the code ’s appropriateness and correctness, with a range of scores allowed.
#     """,
#     "prompt_code": """
#     As a programming instructor , evaluate the student code. Question: {df["question"]. iloc[i]} Solution: {df["response"]. iloc[i]}, give the score only. The ouput be:- Score:\
#     """
# }

essay_evaluating = {
    "Generation": "You are an expert writing instructor responsible for evaluating student essays. Write five separate essay responses that reflect scores from 0 to 4 out of 5 for the following essay prompt: {question}. Use the provided model answer as a reference: {Solution}. Each response should demonstrate the typical characteristics and quality of writing expected at its respective score level.",
    "Evaluating": "You are an expert writing instructor who evaluates student essay submissions. I want you to evaluate the essays from students in this file, where the column {questionsummary} contains the essay prompt and the column {responsesummary} contains the student's essay response.\n\nInstruction:\n- Grade each response out of 5 (0, 1, 2, 3, 4, 5)\n- Evaluate based on how well the essay addresses the prompt\n- Do not focus on grammar, spelling, or stylistic polish unless it significantly affects understanding.",
    "system_prompt": "As a writing instructor, you've been tasked with assessing student essay assignments. In the provided file, each entry contains an essay question (in the column {question}) and a corresponding essay written by a student (in the column {response}). Your job is to evaluate how well the essay addresses the essay prompt. Assign scores from 0 to 5, considering the essay’s relevance, clarity, structure, and completeness in addressing the topic. Provide scores that accurately reflect the essay’s quality and adherence to the prompt.",
    "prompt_code": "As a writing instructor, evaluate the student essay.\n Question:\n{question}\n\nEssay:\n{response}\n Provide the score only in the format:- Score:"
}

class BeGradingScorer:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name: str, question: str, solution: str, category: str = "Evaluating") -> dict:
        match category:
            case "Evaluating":
                prompt = essay_evaluating["Evaluating"]
                prompt = f"{prompt}\n\nQuestion:\n {question}\n\nSolution:\n {solution}"
                messages = [
                    {"role": "user", "content": prompt}
                ]
            case "Generation":
                prompt = essay_evaluating["Generation"]
                prompt = f"{prompt}\n\nquestionsummary:\n {question}\n\nresponsesummary:\” {solution}"
                messages = [
                    {"role": "user", "content": prompt}
                ]
            case "Scoring":
                system_prompt = essay_evaluating["system_prompt"]
                prompt = essay_evaluating["prompt_code"]
                prompt = prompt.replace("{question}", question).replace("{response}", solution)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            case _:
                raise ValueError("Invalid category specified. Use 'Code Evaluation' or 'Essay Evaluation'.")
        output_llm_code = {
            "messages": messages,
            "category": category,
            "task": question,
            "model_name": model_name,
            "essay": solution
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_code = result["choices"][0]["message"]["content"]
        output_llm_code["output"] = output_code
        json_llm_string = json.dumps(result)
        output_llm_code["result"] = json_llm_string
        reason = ""
        output_score = -100
        try:
            if "</think>" in output_code:
                reason = output_code.split("</think>")[0].strip()
                reason = f"{reason}\n</think>"
                output_score = output_code.split("</think>")[-1].strip().split("Score:")[-1].strip()
            else:
                reason = output_code.split("Score:")[0].strip()
                output_score = output_code.split("Score:")[-1].strip()
            output_score = float(output_score)
        except Exception as e:
            print(f"Error parsing score: {e}")
        output_llm_code["score"] = output_score
        output_llm_code["reason"] = reason.strip()
        return output_llm_code


if __name__ == '__main__':
    prompt_i = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends.\nWrite a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
    essay_i = "Dear Editor,\nIn today’s fast-paced world, computers have become an essential part of our daily lives. While some people worry that computers might be doing more harm than good, I firmly believe that computers have a positive impact on society and can help us grow intellectually, socially, and even emotionally—if used wisely.\nFirst and foremost, computers are powerful educational tools. With just a few clicks, students and adults alike can access a world of knowledge. From learning new languages to exploring the history of ancient civilizations, computers allow us to go beyond textbooks and classrooms. This ability to explore and learn on our own empowers people and encourages lifelong learning.\nComputers also help build skills that are important in today’s job market. Typing, researching, and using various software programs are now basic requirements for many careers. Even more, video games and creative programs help improve hand-eye coordination and problem-solving abilities, especially among children and teens.\nAdditionally, computers bring people closer together. Through video calls, emails, and social media, families and friends can stay in touch no matter how far apart they are. In a world where people often move for work or school, this kind of connection is priceless.\nOf course, it’s true that we need balance. Too much screen time isn’t healthy. But the solution isn’t to avoid computers—it’s to use them responsibly. Just like with food, books, or anything else, moderation is key.\nLet’s not fear technology. Instead, let’s teach ourselves and our children how to use computers in smart, productive ways. When used thoughtfully, computers don’t pull us away from life—they help us engage with it more fully.\nSincerely,"
    link = "anduin.hucompute.org"
    model_name_test = "test:DeepSeek-R1"
    port_i = 11434
    seed_i = 42
    temperature_i = 1.0
    # category_i = "Evaluating"  # or "Generation", "Scoring"
    category_i = "Scoring"
    scorer = BeGradingScorer(link, port_i, seed_i, temperature_i)
    output = scorer.run_message(model_name=model_name_test, question=prompt_i, solution=essay_i, category=category_i)
    print(output)



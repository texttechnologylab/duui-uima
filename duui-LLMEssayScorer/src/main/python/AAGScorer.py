from EssayScorer import OpenAIProcessing
import json
# pormpt is from the Paper: https://arxiv.org/pdf/2501.14305
# Background is the system prompt of the Model
# Question is the question to be answered by the student
# Student Answer is the student's response to the question
# Marking Scheme is the rubric used to evaluate the student's response
prompt_template = """
Overview: Your role is to assess and provide feedback on a student's response to a specific task in a <Course> assignment. Each task contains multiple components, and your are required to focus on evaluating the last question.
Background: <Question Background>
Question: <Question>
Student's Response: <Student Answer>
Marking Scheme: <Marking Scheme>

Instructions for the Model:
    - Focus: Grade and provide comments exclusively on the last question.
    - Scoring: Rate the student's answer from 1 to 10 according to the provided marking scheme.
    - Feedback: Provide a brief and constructive critique. Highlighting strengths and areas for improvement according to the marking scheme. Feedback must be concise.
"""

background_template = "As a virtual evaluator with expertise in <Course>, your role is to critically analyze and grade student responses based on the provided marking scheme."
rubric_template = """Output as Json:
{
    "score": <Score Point from 1 to 10>,
    "feedback": "<Brief and constructive feedback>",
    "strengths": "<Strengths of the response according to the rubric>",
    "areas_for_improvement": "<Areas for improvement according to the rubric>"
}
Give only the JSON output without any additional text.
"""


class AGGScore:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name: str, course:str, question: str, student_answer: str) -> dict:
        messages = [
            {"role": "user", "content": prompt_template.replace("<Course>", course).replace("<Question Background>", background_template).replace("<Question>", question).replace("<Student Answer>", student_answer).replace("<Marking Scheme>", rubric_template)}
        ]
        output_llm_agg = {
            "messages": messages,
            "background": background_template.replace("<Course>", course),
            "category": "AGG",
            "task": question,
            "student_answer": student_answer,
            "marking_scheme": rubric_template,
            "model_name": model_name
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_agg = result["choices"][0]["message"]["content"]
        output_llm_agg["output"] = output_agg
        output_llm_agg["result"] = result
        try:
            first_index = output_agg.find('{')
            last_index = output_agg.rfind('}')
            json_string = output_agg[first_index:last_index + 1]
            output_llm_agg["json_output"] = json_string
            json_dict = json.loads(json_string)
            output_llm_agg["score"] = json_dict.get("score")
            output_llm_agg["reason"] = json_dict.get("feedback")
        except Exception as e:
            print(f"Error parsing JSON output: {e}")
            output_llm_agg["json_output"] = None
            output_llm_agg["score"] = -1
            output_llm_agg["reason"] = "Failed to parse JSON output from the model response."
        return output_llm_agg


if __name__ == '__main__':
    link = "127.0.0.1"
    model_name_test = "test:DeepSeek-R1"
    port_i = 11434
    seed_i = 42
    temperature_i = 1.0
    course_i = "English Composition"
    background_i = "As a virtual evaluator with expertise in English composition, your role is to critically analyze and grade student essays."
    prompt_i = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends.\nWrite a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
    essay_i = "Dear Editor,\nIn today’s fast-paced world, computers have become an essential part of our daily lives. While some people worry that computers might be doing more harm than good, I firmly believe that computers have a positive impact on society and can help us grow intellectually, socially, and even emotionally—if used wisely.\nFirst and foremost, computers are powerful educational tools. With just a few clicks, students and adults alike can access a world of knowledge. From learning new languages to exploring the history of ancient civilizations, computers allow us to go beyond textbooks and classrooms. This ability to explore and learn on our own empowers people and encourages lifelong learning.\nComputers also help build skills that are important in today’s job market. Typing, researching, and using various software programs are now basic requirements for many careers. Even more, video games and creative programs help improve hand-eye coordination and problem-solving abilities, especially among children and teens.\nAdditionally, computers bring people closer together. Through video calls, emails, and social media, families and friends can stay in touch no matter how far apart they are. In a world where people often move for work or school, this kind of connection is priceless.\nOf course, it’s true that we need balance. Too much screen time isn’t healthy. But the solution isn’t to avoid computers—it’s to use them responsibly. Just like with food, books, or anything else, moderation is key.\nLet’s not fear technology. Instead, let’s teach ourselves and our children how to use computers in smart, productive ways. When used thoughtfully, computers don’t pull us away from life—they help us engage with it more fully.\nSincerely,"
    set_1_rubrics = """
    Output as Json:
    {
        "score": <Score Point from 1 to 10>,
        "feedback": "<Brief and constructive feedback>",
        "strengths": "<Strengths of the response according to the rubric>",
        "areas_for_improvement": "<Areas for improvement according to the rubric>"
    }
    Give only the JSON output without any additional text.
    """
    # set_1_rubrics = """
    # Score Point 1: An undeveloped response that may take a position but offers no more than very minimal support. Typical elements:
    # - Contains few or vague details.
    # - Is awkward and fragmented.
    # - May be difficult to read and understand.
    # - May show no awareness of audience.
    #
    # Score Point 2: An under-developed response that may or may not take a position. Typical elements:
    # - Contains only general reasons with unelaborated and/or list-like details.
    # - Shows little or no evidence of organization.
    # - May be awkward and confused or simplistic.
    # - May show little awareness of audience.
    #
    # Score Point 3: A minimally-developed response that may take a position, but with inadequate support and details. Typical elements:
    # - Has reasons with minimal elaboration and more general than specific details.
    # - Shows some organization.
    # - May be awkward in parts with few transitions.
    # - Shows some awareness of audience.
    #
    # Score Point 4: A somewhat-developed response that takes a position and provides adequate support. Typical elements:
    # - Has adequately elaborated reasons with a mix of general and specific details.
    # - Shows satisfactory organization.
    # - May be somewhat fluent with some transitional language.
    # - Shows adequate awareness of audience.
    #
    # Score Point 5: A developed response that takes a clear position and provides reasonably persuasive support. Typical elements:
    # - Has moderately well elaborated reasons with mostly specific details.
    # - Exhibits generally strong organization.
    # - May be moderately fluent with transitional language throughout.
    # - May show a consistent awareness of audience.
    #
    # Score Point 6: A well-developed response that takes a clear and thoughtful position and provides persuasive support. Typical elements:
    # - Has fully elaborated reasons with specific details.
    # - Exhibits strong organization.
    # - Is fluent and uses sophisticated transitional language.
    # - May show a heightened awareness of audience."""

    scorer = AGGScore(link, port_i, seed_i, temperature_i)
    result_i = scorer.run_message(model_name_test, course_i, background_i, prompt_i, essay_i, set_1_rubrics)
    print(result_i)


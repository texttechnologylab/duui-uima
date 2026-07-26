from EssayScorer import OpenAIProcessing
# Normaly filled with medical however this placeholder makes more flexible for other uses
# Second Placeholder is for maximum points for grading
system_prompt = "You are a {} teacher that needs to grade questions. Compare the answer given by the student to the key and return a grad between 0 and max. {} points. Decimal grades are allowed. Only return the grade, nothing else!"
question = "Question: {}"
#  key (rubric or sample solution)
key_user = "Key: {}"
answer = "Student's answer: {}"

class GradingMedicalEducationScorer:
    def __init__(self, url: str, port: int, seed: int, temperature: float, api_key: str = None):
        self.openai = OpenAIProcessing(url, port, seed, temperature, api_key)

    def run_message(self, model_name: str, course: str, question_text: str, key_text: str, student_answer: str, max_points: int=10) -> dict:
        messages = [
            {"role": "system", "content": system_prompt.format(course, max_points)},
            {"role": "user", "content": question.format(question_text)},
            {"role": "user", "content": key_user.format(key_text)},
            {"role": "user", "content": answer.format(student_answer)}
        ]
        output_llm_grading = {
            "messages": messages,
            "category": course,
            "task": question_text,
            "student_answer": student_answer,
            "key": key_text,
            "model_name": model_name
        }
        result = self.openai.process_messages(model_name=model_name, messages=messages)
        output_grading = result["choices"][0]["message"]["content"]
        output_llm_grading["output"] = output_grading
        output_llm_grading["result"] = result
        try:
            grade_text = output_grading.split("\n")[-1].split(":")[-1].strip()
            grade = float(grade_text)
            reason = "Grade extracted successfully."
        except Exception as e:
            print(f"Error parsing grade output: {e}")
            grade = -1
            reason = "Failed to parse grade from the model response."
        output_llm_grading["score"] = grade
        output_llm_grading["reason"] = reason
        return output_llm_grading


if __name__ == '__main__':
    prompt_i = "More and more people use computers, but not everyone agrees that this benefits society. Those who support advances in technology believe that computers have a positive effect on people. They teach hand-eye coordination, give people the ability to learn about faraway places and people, and even allow people to talk online with other people. Others have different ideas. Some experts are concerned that people are spending too much time on their computers and less time exercising, enjoying nature, and interacting with family and friends.\nWrite a letter to your local newspaper in which you state your opinion on the effects computers have on people. Persuade the readers to agree with you."
    essay_i = "Dear Editor,\nIn today’s fast-paced world, computers have become an essential part of our daily lives. While some people worry that computers might be doing more harm than good, I firmly believe that computers have a positive impact on society and can help us grow intellectually, socially, and even emotionally—if used wisely.\nFirst and foremost, computers are powerful educational tools. With just a few clicks, students and adults alike can access a world of knowledge. From learning new languages to exploring the history of ancient civilizations, computers allow us to go beyond textbooks and classrooms. This ability to explore and learn on our own empowers people and encourages lifelong learning.\nComputers also help build skills that are important in today’s job market. Typing, researching, and using various software programs are now basic requirements for many careers. Even more, video games and creative programs help improve hand-eye coordination and problem-solving abilities, especially among children and teens.\nAdditionally, computers bring people closer together. Through video calls, emails, and social media, families and friends can stay in touch no matter how far apart they are. In a world where people often move for work or school, this kind of connection is priceless.\nOf course, it’s true that we need balance. Too much screen time isn’t healthy. But the solution isn’t to avoid computers—it’s to use them responsibly. Just like with food, books, or anything else, moderation is key.\nLet’s not fear technology. Instead, let’s teach ourselves and our children how to use computers in smart, productive ways. When used thoughtfully, computers don’t pull us away from life—they help us engage with it more fully.\nSincerely,"
    link = "localhost"  # Replace with your server address
    course = "English"
    key_rubric = "<Score Point from 1 to 10>"
    model_name_test = "test:DeepSeek-R1"
    port_i = 11434
    seed_i = 42
    temperature_i = 1.0
    scorer = GradingMedicalEducationScorer(link, port_i, seed_i, temperature_i)
    output = scorer.run_message(model_name=model_name_test, course=course, question_text=prompt_i, key_text=key_rubric, student_answer=essay_i, max_points=10)
    print(output)
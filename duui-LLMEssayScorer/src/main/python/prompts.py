from langchain.prompts import PromptTemplate

template_1 = """
{model_prefix}

You are given an essay written by a student and the corresponding prompt for the 7th to 10th grade student. 

#### Prompt: 
'''{prompt}'''

### Task:
{instruction}

#### Student essay: 
'''{essay}'''

Finally, after everything, give the grade in the following json format: 
{format_instructions}

{model_suffix}
"""

prompt_template_1 = PromptTemplate(template=template_1,
                                   input_variables=["instruction", "format_instructions", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

template_2 = """
{model_prefix}

Imagine you are a teacher's assistant in a middle school, tasked with reviewing a 7th to 10th grade student's essay. You have the essay and the prompt that was given to the student.

#### Original Prompt Provided to Student:
'''{prompt}'''

### Review Task:
{instruction}

#### Student's Essay for Review:
'''{essay}'''

After reviewing, provide feedback and a grade using this format:
{format_instructions}

{model_suffix}
"""

prompt_template_2 = PromptTemplate(template=template_2,
                                   input_variables=["instruction", "format_instructions", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

template_3 = """
{model_prefix}

You are part of an educational research team analyzing the writing skills of students in grades 7 to 10. You have been given a student's essay and the prompt they responded to.

#### Essay Prompt:
'''{prompt}'''

### Analysis Task:
{instruction}

#### Analyzed Student Essay:
'''{essay}'''

Conclude your analysis with a grade and comments in the following format:
{format_instructions}

{model_suffix}
"""

prompt_template_3 = PromptTemplate(template=template_3,
                                   input_variables=["instruction", "format_instructions", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

template_4 = """
{model_prefix}

You are a creative writing mentor evaluating a piece written by a student in grades 7 to 10. The student's work is based on a specific prompt.

#### Creative Prompt Given:
'''{prompt}'''

### Critique Instructions:
{instruction}

#### Student's Creative Piece:
'''{essay}'''

End your critique with a structured assessment and grade, as outlined here:
{format_instructions}

{model_suffix}
"""

prompt_template_4 = PromptTemplate(template=template_4,
                                   input_variables=["instruction", "format_instructions", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

holistic_scoring_instruction1 = """
Given this essay that was written for the given prompt, grade the essay using those ranges: {scoring_instructions} 
"""

holistic_scoring_instruction2 = """
Review the provided essay in response to the given prompt. Assess its quality and assign a grade according to the following criteria: {scoring_instructions}.
"""

holistic_scoring_instruction3 = """
Examine the essay written in response to the specified prompt. Utilize the following grading ranges to evaluate and score the essay: {scoring_instructions}.
"""

holistic_scoring_instruction4 = """
Analyze the submitted essay that corresponds to the given prompt. Apply these grading standards to determine its score: {scoring_instructions}.
"""

holistic_scoring_instructions = [holistic_scoring_instruction1, holistic_scoring_instruction2,
                                    holistic_scoring_instruction3, holistic_scoring_instruction4]

holistic_scoring_instruction_with_small_rubric1 = """
Grade the given essay using the following rubric: 
{rubric_small}

Use those score ranges: {scoring_instructions}
"""

holistic_scoring_instruction_with_small_rubric2 = """
Please evaluate the provided essay based on the criteria outlined in this rubric: 
{rubric_small}

For grading, apply these scoring ranges: {scoring_instructions}
"""

holistic_scoring_instruction_with_small_rubric3 = """
Assess the essay using the criteria specified below: 
{rubric_small}

Assign a score to the essay within these ranges: {scoring_instructions}
"""

holistic_scoring_instruction_with_small_rubric4 = """
Review the essay according to the following evaluation guidelines: 
{rubric_small}

Use the following scoring brackets for your assessment: {scoring_instructions}
"""

holistic_scoring_instructions_with_small_rubric = [holistic_scoring_instruction_with_small_rubric1,
                                                   holistic_scoring_instruction_with_small_rubric2,
                                                   holistic_scoring_instruction_with_small_rubric3,
                                                   holistic_scoring_instruction_with_small_rubric4]

holistic_scoring_instruction_with_full_rubric1 = """
Grade the given essay using the following rubric:
{rubric_full}

Use those score ranges: {scoring_instructions}
"""

holistic_scoring_instruction_with_full_rubric2 = """
Please evaluate the submitted essay according to the criteria outlined in the rubric below:
{rubric_full}

Assign a score to the essay within the ranges specified here:
{scoring_instructions}
"""

holistic_scoring_instruction_with_full_rubric3 = """
Using the detailed rubric provided, assess the quality of the provided essay:
{rubric_full}

Based on your assessment, please allocate a score to the essay according to these scoring guidelines:
{scoring_instructions}
"""

holistic_scoring_instruction_with_full_rubric4 = """
Carefully review the essay in light of the following grading rubric:
{rubric_full}

Upon review, allocate a score to the essay within the following scoring parameters:
{scoring_instructions}
"""

holistic_scoring_instructions_with_full_rubric = [holistic_scoring_instruction_with_full_rubric1,
                                                    holistic_scoring_instruction_with_full_rubric2,
                                                    holistic_scoring_instruction_with_full_rubric3,
                                                    holistic_scoring_instruction_with_full_rubric4]

explanation_somewhere_instruction1 = """
Grade the given essay using the following rubric: 
{rubric_small}

Use those score ranges: {scoring_instructions}

Provide an explanation for your score as well. 
"""

explanation_somewhere_instruction2 = """
Please assess the submitted essay according to the criteria outlined in this rubric: 
{rubric_small}

Scores should be allocated based on these guidelines: {scoring_instructions}

Additionally, include a detailed rationale for the score you assign.
"""

explanation_somewhere_instruction3 = """
Evaluate the provided essay by referring to the standards specified here: 
{rubric_small}

Utilize the following scoring range for your evaluation: {scoring_instructions}

Also, furnish a comprehensive justification for the grade you determine.
"""

explanation_somewhere_instruction4 = """
Rate the essay in front of you using these evaluation criteria: 
{rubric_small}

Your scoring should align with these parameters: {scoring_instructions}

Please also give a thorough explanation to support the score you decide upon.
"""

explanation_somewhere_instructions = [explanation_somewhere_instruction1, explanation_somewhere_instruction2,
                                        explanation_somewhere_instruction3, explanation_somewhere_instruction4]

explain_first_instruction1 = """
Analyse the given essay using the following rubric: 
{rubric_small}. 

To do this, first explain using the scoring rubric why you chose the score. After you analysed the essay, give a final grade in json format.
"""

explain_first_instruction2 = """
Utilize the provided scoring rubric ({rubric_small}) to evaluate the essay. Begin by detailing the reasons for your assigned score based on the rubric's criteria. Conclude by summarizing your analysis with a final grade, presented in json format.
"""

explain_first_instruction3 = """
Apply the scoring guidelines from {rubric_small} to assess the essay. Start by discussing how the essay meets or falls short of each criterion in the rubric. Finalize your assessment with a grade, formatted in json.
"""

explain_first_instruction4 = """
Use the scoring rubric ({rubric_small}) as a basis to critically analyze the essay. Explain how the essay aligns with each aspect of the rubric, justifying your evaluation. Conclude with a definitive grade, expressed in json format.
"""

explain_first_instructions = [explain_first_instruction1, explain_first_instruction2, explain_first_instruction3,
                                explain_first_instruction4]

feedback_somewhere_instruction1 = """
Grade the given essay using the following rubric:
{rubric_small}

Use those score ranges: {scoring_instructions}

Provide comprehensive feedback for the student that helps them to achieve better grades in the future.
"""

feedback_somewhere_instruction2 = """
Please evaluate the essay in accordance with the criteria outlined in: {rubric_small}. 
Assign a grade based on these standards: {scoring_instructions}. 
Offer detailed and constructive feedback to assist the student in improving their writing skills for future assignments.
"""

feedback_somewhere_instruction3 = """
Utilize the provided rubric ({rubric_small}) to assess the essay. 
Grade it according to these parameters: {scoring_instructions}. 
Your feedback should be thorough, focusing on areas of strength and suggesting improvements to help the student enhance their academic writing.
"""

feedback_somewhere_instruction4 = """
Conduct an assessment of the submitted essay using this specific rubric: {rubric_small}. 
Apply the grading criteria as per these guidelines: {scoring_instructions}. 
Your feedback should be insightful and supportive, guiding the student towards achieving higher grades in their future essays.
"""

feedback_somewhere_instructions = [feedback_somewhere_instruction1, feedback_somewhere_instruction2,
                                    feedback_somewhere_instruction3, feedback_somewhere_instruction4]

feedback_first_instruction1 = """
Analyse the given essay using the following rubric:
{rubric_small}

Use those score ranges: {scoring_instructions}

To do this, first provide comprehensive feedback for the student that helps them to achieve better grades in the future.
Then give the final score.
"""

feedback_first_instruction2 = """
Begin by carefully reviewing the submitted essay in light of the criteria outlined in {rubric_small}. 
After your thorough analysis, offer detailed and constructive feedback aimed at guiding the student towards academic improvement. 
Conclude your review by assigning a score to the essay, adhering to the guidelines specified in {scoring_instructions}.
"""

feedback_first_instruction3 = """
First, evaluate the essay against the criteria mentioned in {rubric_small}. 
Your evaluation should include specific, actionable suggestions for the student to enhance their writing skills and essay quality. 
Following your comprehensive feedback, assign a score to the essay based on the scale provided in {scoring_instructions}.
"""

feedback_first_instruction4 = """
Commence your assessment by applying the criteria from {rubric_small} to the essay. 
Focus on delivering in-depth feedback that is both informative and beneficial for the student's future academic endeavors. 
After providing this feedback, conclude by scoring the essay as per the range defined in {scoring_instructions}.
"""

feedback_first_instructions = [feedback_first_instruction1, feedback_first_instruction2, feedback_first_instruction3,
                                feedback_first_instruction4]

feedback_and_explanation_instruction1 = """
Analyse the given essay using the following rubric:
{rubric_small}

Use those score ranges: {scoring_instructions}

To do this, first explain the different parts of the rubric in relation to the text and which score you would give for each part.
Then give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future.
Finally, provide the final score.
"""

feedback_and_explanation_instruction2 = """
First, provide a detailed breakdown of the essay based on the criteria in {rubric_small}. Assign scores to each section of the rubric according to {scoring_instructions}.
After scoring, offer a comprehensive critique of the essay. Highlight the strengths and weaknesses as per the rubric's criteria.
Conclude with actionable advice for the student to improve their writing, focusing on the areas where they lost points.
Summarize by presenting the total score from the rubric.
"""

feedback_and_explanation_instruction3 = """
Begin by correlating the sections of the essay with the elements in {rubric_small}. Evaluate each element and assign a score as per {scoring_instructions}.
Next, construct a feedback report for the student. In this report, pinpoint specific areas where the essay excelled or fell short, referring to the rubric's criteria.
Offer constructive suggestions for the student to enhance their writing skills, particularly in weaker areas.
End with the calculation of the overall score based on the rubric.
"""

feedback_and_explanation_instruction4 = """
Commence with an analysis of the essay, aligning it with the components outlined in {rubric_small}. Rate each component using the guidelines provided in {scoring_instructions}.
Proceed to draft a feedback letter to the student. This letter should clearly outline where the essay meets or deviates from the rubric standards.
Provide targeted advice for improvement, focusing on the rubric areas where the essay could score better.
Wrap up with a final score, tallying the individual component scores as per the rubric.
"""

feedback_and_explanation_instructions = [feedback_and_explanation_instruction1,
                                            feedback_and_explanation_instruction2,
                                            feedback_and_explanation_instruction3,
                                            feedback_and_explanation_instruction4]

persona_instruction = """
Act as an experienced junior high school teacher. You are very experienced in grading essays and have a lot of empathy for the students. Analyse the given essay using the following rubric:
{rubric_small}

Use those score ranges: {scoring_instructions}

First, provide feedback for the students that explains their mistakes and errors and additionally gives them tips to avoid them in the future. Finally, give the final score in the required score range and format.
"""

persona_instructions = [persona_instruction]

chain_of_thought_instruction1 = """
Analyse the given essay using the following rubric and give helpful feedback to the student:
{rubric_small}

Use those score ranges: {scoring_instructions}

Lets think step by step. Make sure to output the score in json format at the only at the end.
"""

chain_of_thought_instruction2 = """
Please evaluate the provided essay according to this specific rubric:
{rubric_small}

Scores should be assigned based on these criteria: {scoring_instructions}

Proceed methodically through each step. Conclude your analysis by presenting the final score in a json format.
"""

chain_of_thought_instruction3 = """
Conduct a thorough assessment of the essay using the rubric below:
{rubric_small}

Adhere to the following scoring guidelines: {scoring_instructions}

Break down your analysis into clear steps. Ensure the final score is given in json format at the end of your evaluation.
"""

chain_of_thought_instruction4 = """
Examine the student's essay in detail, utilizing the rubric provided:
{rubric_small}

Apply these scoring ranges for evaluation: {scoring_instructions}

Tackle the analysis in a step-by-step manner. The score should be formatted in json and presented at the conclusion of your feedback.
"""

chain_of_thought_instructions = [chain_of_thought_instruction1, chain_of_thought_instruction2,
                                    chain_of_thought_instruction3, chain_of_thought_instruction4]

chain_of_thought_detailed_instruction1 = """
Analyse the given essay using the following rubric and give helpful feedback to the student:
{rubric_small}

Use those score ranges: {scoring_instructions}

Lets think step by step. 
First, analyse the quality of the essay in terms of the given rubric.
Then, give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future.
As a final step, output the score in the json format. Make sure to output this only at the end.
"""

chain_of_thought_detailed_instruction2 = """
Begin by evaluating the essay based on the criteria outlined in the rubric: {rubric_small}. 
Consider the scoring guidelines provided: {scoring_instructions}. 
First, conduct a thorough analysis of the essay according to the rubric standards. 
Next, provide constructive feedback to the student, highlighting areas for improvement and suggesting strategies to enhance their writing skills. 
Conclude with a summary of the essay's strengths and weaknesses. 
Finally, present the essay's score in a json format at the end of your analysis.
"""

chain_of_thought_detailed_instruction3 = """
Follow these steps to assess the student's essay: 
First, reference the provided rubric: {rubric_small}, and apply it to evaluate the essay. 
Use the scoring ranges given: {scoring_instructions} for accurate assessment. 
Provide detailed feedback to the student, pinpointing specific areas of the essay that align or deviate from the rubric, along with advice for future improvement. 
Your feedback should be clear, constructive, and actionable. 
After your comprehensive review, conclude by outputting the final score in json format, ensuring this is done only at the very end.
"""

chain_of_thought_detailed_instruction4 = """
To evaluate the student's essay, proceed as follows:
Start with the provided rubric: {rubric_small}, to assess the essay's attributes. 
Adhere to the scoring guidelines: {scoring_instructions} for consistency. 
Your analysis should first focus on how well the essay meets the criteria in the rubric. 
Then, craft feedback for the student that is both informative and helpful, addressing any shortcomings and providing practical advice for future essays. 
The feedback should be encouraging yet honest. 
Conclude your evaluation by scoring the essay, presented in json format at the conclusion of your feedback.
"""

chain_of_thought_detailed_instructions = [chain_of_thought_detailed_instruction1,
                                            chain_of_thought_detailed_instruction2,
                                            chain_of_thought_detailed_instruction3,
                                            chain_of_thought_detailed_instruction4]

one_shot_instruction1 = """
Analyse the given essay using the following rubric and give helpful feedback to the student:
{rubric_small}

An example for the scoring would be: 
{examples}

Use those score ranges: {scoring_instructions}

To do this, first explain the different parts of the rubric in relation to the text and which score you would give for each part.
Then give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future.
Finally, provide the final score.
"""

one_shot_instruction2 = """
Begin by breaking down the essay according to the criteria in the rubric provided here:
{rubric_small}

Reference this example for clarity on how to assign scores:
{examples}

Scores should be allocated based on these guidelines:
{scoring_instructions}

Firstly, correlate each aspect of the rubric to specific parts of the essay and assign a score for each. Next, offer constructive feedback to the student, highlighting both strengths and areas for improvement, with suggestions on how to enhance their writing skills. Conclude by summarizing the overall performance with a final score.
"""

one_shot_instruction3 = """
To evaluate the essay, use the following assessment criteria:
{rubric_small}

For an understanding of how to apply these criteria, refer to:
{examples}

Adhere to these scoring parameters:
{scoring_instructions}

Your analysis should start by associating elements of the rubric with the essay's content, providing a score for each category. Following this, present feedback that not only identifies errors but also guides the student on how to rectify these issues in future essays. Finish by calculating and presenting the total score.
"""

one_shot_instruction4 = """
Review the student's essay using this evaluation framework:
{rubric_small}

Consider this scoring example for guidance:
{examples}

Apply these scoring ranges in your evaluation:
{scoring_instructions}

Commence by examining each rubric component in the context of the essay, assigning a score for each. Proceed to offer detailed feedback to the student, pinpointing specific mistakes and providing actionable advice for improvement. Conclude by tallying the scores to arrive at a final grade.
"""

one_shot_instructions = [one_shot_instruction1, one_shot_instruction2, one_shot_instruction3, one_shot_instruction4]

few_shot_instruction1 = """
Analyse the given essay using the following rubric and give helpful feedback to the student:
{rubric_small}

Some examples for the scoring would be: 
{examples}

Use those score ranges: {scoring_instructions}

To do this, first explain the different parts of the rubric in relation to the text and which score you would give for each part.
Then give feedback to the student that explains their mistakes and errors and additionally gives them tips to avoid them in the future.
Finally, provide the final score.
"""

few_shot_instruction2 = """
Begin by breaking down the essay according to the criteria in the rubric provided here:
{rubric_small}

Reference these examples for clarity on how to assign scores:
{examples}

Scores should be allocated based on these guidelines:
{scoring_instructions}

Firstly, correlate each aspect of the rubric to specific parts of the essay and assign a score for each. Next, offer constructive feedback to the student, highlighting both strengths and areas for improvement, with suggestions on how to enhance their writing skills. Conclude by summarizing the overall performance with a final score.
"""

few_shot_instruction3 = """
To evaluate the essay, use the following assessment criteria:
{rubric_small}

For an understanding of how to apply these criteria, refer to:
{examples}

Adhere to these scoring parameters:
{scoring_instructions}

Your analysis should start by associating elements of the rubric with the essay's content, providing a score for each category. Following this, present feedback that not only identifies errors but also guides the student on how to rectify these issues in future essays. Finish by calculating and presenting the total score.
"""

few_shot_instruction4 = """
Review the student's essay using this evaluation framework:
{rubric_small}

Consider these scoring examples for guidance:
{examples}

Apply these scoring ranges in your evaluation:
{scoring_instructions}

Commence by examining each rubric component in the context of the essay, assigning a score for each. Proceed to offer detailed feedback to the student, pinpointing specific mistakes and providing actionable advice for improvement. Conclude by tallying the scores to arrive at a final grade.
"""

few_shot_instructions = [few_shot_instruction1, few_shot_instruction2, few_shot_instruction3, few_shot_instruction4]

feedback_only_instruction1 = """
Analyse the given essay using the following rubric:
{rubric_small}

Provide comprehensive feedback for the student that helps them to achieve better grades in the future.
"""

feedback_only_instruction2 = """
Please evaluate the essay in accordance with the criteria outlined in: {rubric_small}. 
Offer detailed and constructive feedback to assist the student in improving their writing skills for future assignments.
"""

feedback_only_instruction3 = """
Utilize the provided rubric ({rubric_small}) to assess the essay. 
Your feedback should be thorough, focusing on areas of strength and suggesting improvements to help the student enhance their academic writing.
"""

feedback_only_instruction4 = """
Conduct an assessment of the submitted essay using this specific rubric: {rubric_small}. 
Your feedback should be insightful and supportive, guiding the student towards achieving higher grades in their future essays.
"""

feedback_only_instructions = [feedback_only_instruction1, feedback_only_instruction2, feedback_only_instruction3,
                                feedback_only_instruction4]

scoreless_template_1 = """
{model_prefix}

You are given an essay written by a student and the corresponding prompt for the 7th to 10th grade student. 

#### Prompt: 
'''{prompt}'''

### Task:
{instruction}

#### Student essay: 
'''{essay}'''

{model_suffix}
"""

scoreless_prompt_template_1 = PromptTemplate(template=scoreless_template_1,
                                   input_variables=["instruction", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

scoreless_template_2 = """
{model_prefix}

Imagine you are a teacher's assistant in a middle school, tasked with reviewing a 7th to 10th grade student's essay. You have the essay and the prompt that was given to the student.

#### Original Prompt Provided to Student:
'''{prompt}'''

### Review Task:
{instruction}

#### Student's Essay for Review:
'''{essay}'''

{model_suffix}
"""

scoreless_prompt_template_2 = PromptTemplate(template=scoreless_template_2,
                                   input_variables=["instruction", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

scoreless_template_3 = """
{model_prefix}

You are part of an educational research team analyzing the writing skills of students in grades 7 to 10. You have been given a student's essay and the prompt they responded to.

#### Essay Prompt:
'''{prompt}'''

### Analysis Task:
{instruction}

#### Analyzed Student Essay:
'''{essay}'''

{model_suffix}
"""

scoreless_prompt_template_3 = PromptTemplate(template=scoreless_template_3,
                                   input_variables=["instruction", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])

scoreless_template_4 = """
{model_prefix}

You are a creative writing mentor evaluating a piece written by a student in grades 7 to 10. The student's work is based on a specific prompt.

#### Creative Prompt Given:
'''{prompt}'''

### Critique Instructions:
{instruction}

#### Student's Creative Piece:
'''{essay}'''

{model_suffix}
"""

scoreless_prompt_template_4 = PromptTemplate(template=scoreless_template_4,
                                   input_variables=["instruction", "essay", "prompt",
                                                    "model_prefix", "model_suffix"])
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Union
import logging
from fastapi import FastAPI, Response
from cassis import load_typesystem
from functools import lru_cache
from threading import Lock
from starlette.responses import PlainTextResponse
from EssayScorer import EssayScorer
from BeGradingScorer import BeGradingScorer
from LLMAESScorer import ScoreSlowStudent,ScoreStudent,all_rubrics
from GradeLikeHuman import GradingLikeHumanScorer
from EssayEvalScoring import EssayEvalScoring
from AAGScorer import AGGScore
from GradingMedicalEducation import GradingMedicalEducationScorer
from AESMTSScorer import Vanilla_OpenAI, MTS_OpenAI
import json
import time
import torch
model_lock = Lock()

class UimaSentence(BaseModel):
    text: str
    begin: int
    end: int
    typeName: str


class UimaSentenceSelection(BaseModel):
    sentences: List[UimaSentence]

class Settings(BaseSettings):
    # Name of this annotator
    annotator_name: str
    # Version of this annotator
    annotator_version: str
    # Log level
    log_level: str
    # model_name
    model_name: str
    # model_spec_name
    model_spec_name: str
    # # Name of this annotator
    model_version: str
    # #cach_size
    model_cache_size: int
    # # url of the model
    model_source: str
    # # language of the model
    model_lang: str
    # Optional: API key for OpenAI


# Load settings from env vars
settings = Settings()
lru_cache_with_size = lru_cache(maxsize=settings.model_cache_size)
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


# Load the predefined typesystem that is needed for this annotator to work
typesystem_filename = 'TypeSystemEssayScorer.xml'
logger.debug("Loading typesystem from \"%s\"", typesystem_filename)
with open(typesystem_filename, 'rb') as f:
    typesystem = load_typesystem(f)
    logger.debug("Base typesystem:")
    logger.debug(typesystem.to_xml())

# Load the Lua communication script
lua_communication_script_filename = "duui_essayscorer.lua"
logger.debug("Loading Lua communication script from \"%s\"", lua_communication_script_filename)


# Request sent by DUUI
# Note, this is transformed by the Lua script
class DUUIRequest(BaseModel):
    # The texts language
    doc_len: int
    #
    lang: str
    #
    scenarios: list
    #
    scenario_ids: list
    #
    questions:  list
    #
    question_ids: list
    #
    answers: list
    #
    answer_ids : list
    #
    seed: int = None
    #
    temperature: float = None
    #
    url: str
    #
    port: int
    #
    model_llm: str
    # The model name to use for scoring
    name_model: str




# UIMA type: mark modification of the document
class DocumentModification(BaseModel):
    user: str
    timestamp: int
    comment: str


# UIMA type: adds metadata to each annotation
class AnnotationMeta(BaseModel):
    name: str
    version: str
    modelName: str
    modelVersion: str

def fix_unicode_problems(text):
    # fix emoji in python string and prevent json error on response
    # File "/usr/local/lib/python3.8/site-packages/starlette/responses.py", line 190, in render
    # UnicodeEncodeError: 'utf-8' codec can't encode characters in position xx-yy: surrogates not allowed
    clean_text = text.encode('utf-16', 'surrogatepass').decode('utf-16', 'surrogateescape')
    return clean_text

def process_selection(request, model_name: str) -> Dict[str, Union[List[int], List[str], List[float]]]:
    begin = []
    end = []
    results_out = []
    factors = []
    definitions = []
    question_ids = []
    answer_ids = []
    scene_ids = []
    contents = []
    responses = []
    additional = []
    reasons = []
    nameLLMModel=[]

    # Load the model
    with model_lock:
        # Process each question-answer pair
        all_answers = [
            s["text"] for s in request.answers
        ]
        all_questions = [
            s["text"] for s in request.questions
        ]
        all_scenes = []
        if (len(request.scenarios) > 0):
            all_scenes = [
                s["text"] for s in request.scenarios
            ]
        logger.info("Processing %d questions and %d answers with model %s", len(all_questions), len(all_answers), model_name)
        model_spec_name = settings.model_spec_name
        logger.info(f"Model spec name: {model_spec_name}")
        match model_name:
            case "KevSun/Engessay_grading_ML":
                model = load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
                outputs = model.run_messages(all_answers)
                for i, output in enumerate(outputs):
                    for name in output:
                        score = output[name]
                        begin.append(request.answers[i]["begin"])
                        end.append(request.answers[i]["end"])
                        results_out.append(name)
                        factors.append(score)
                        definitions.append("Essay scoring factors")
                        answer_ids.append(request.answer_ids[i])
            case "JacobLinCool/IELTS_essay_scoring_safetensors":
                model = load_model(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
                outputs = model.run_messages(all_answers)
                for i, output in enumerate(outputs):
                    for name in output:
                        score = output[name]
                        begin.append(request.answers[i]["begin"])
                        end.append(request.answers[i]["end"])
                        results_out.append(name)
                        factors.append(score)
                        definitions.append("Essay scoring factors")
                        answer_ids.append(request.answer_ids[i])
            case "BeGradingScorer":
                be_scorer = BeGradingScorer(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = be_scorer.run_message(
                        model_name=request.model_llm,
                        question=question,
                        solution=answer,
                        category="Scoring"
                    )
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    results_out.append("BeGradingScore")
                    factors.append(output["score"])
                    reasons.append(output["reason"])
                    json_llm_string = json.dumps(output)
                    contents.append(output["output"])
                    responses.append(json_llm_string)
                    definitions.append("BeGrading Score")
                    time_seconds = time.time() - start_time
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "LLMAESSlowScorer":
                slow_scorer = ScoreSlowStudent(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = slow_scorer.run_message(
                        model_name=request.model_llm,
                        essay=answer,
                        task=question
                    )
                    time_seconds = time.time() - start_time
                    # content info
                    try:
                        content_info = output["content_info"]
                        results_out.append("LLMAESSlow-ContentScore")
                        factors.append(content_info["score"])
                        reasons.append(content_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing content info: {e}")
                        results_out.append("LLMAESSlow-ContentScore")
                        factors.append(-1.0)
                        reasons.append("Error processing content info")
                    # language info
                    try:
                        language_info = output["language_info"]
                        results_out.append("LLMAESSlow-LanguageScore")
                        factors.append(language_info["score"])
                        reasons.append(language_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing language info: {e}")
                        results_out.append("LLMAESSlow-LanguageScore")
                        factors.append(-1.0)
                        reasons.append("Error processing language info")
                    # structure info
                    try:
                        structure_info = output["structure_info"]
                        results_out.append("LLMAESSlow-StructureScore")
                        factors.append(structure_info["score"])
                        reasons.append(structure_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing structure info: {e}")
                        results_out.append("LLMAESSlow-StructureScore")
                        factors.append(-1.0)
                        reasons.append("Error processing structure info")
                    # overall info
                    try:
                        overall_info = output["overall_info"]
                        results_out.append("LLMAESSlow-OverallScore")
                        factors.append(overall_info["score"])
                        reasons.append(overall_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing overall info: {e}")
                        results_out.append("LLMAESSlow-OverallScore")
                        factors.append(-1.0)
                        reasons.append("Error processing overall info")
                    for _ in range(4):
                        begin.append(request.answers[i]["begin"])
                        end.append(request.answers[i]["end"])
                        json_llm_string = json.dumps(output)
                        contents.append(output["output"])
                        responses.append(json_llm_string)
                        definitions.append("LLMAESSlow Score")
                        additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                        answer_ids.append(request.answer_ids[i])
                        question_ids.append(request.question_ids[i])
                        nameLLMModel.append(request.name_model)
                        if len(all_scenes) > 0:
                            scene_ids.append(request.scenario_ids[i])
            case "LLMAESSScorer":
                llm_scorer = ScoreStudent(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = {}
                    if model_spec_name == "llmaes-scorer-norubrics":
                        output = llm_scorer.run_message(
                            model_name=request.model_llm,
                            set_essay=answer,
                            set_prompt=question,
                            category="zeroshot_norubrics"
                        )
                        logger.info(f"LLMAESSScorer without rubrics")
                    if "llmaes-scorer-zeroshot" in model_spec_name:
                        index_zero = model_spec_name.split("-")[-1]
                        index_zero = int(index_zero) - 1
                        output = llm_scorer.run_message(
                            model_name=request.model_llm,
                            set_essay=answer,
                            set_prompt=question,
                            set_rubrics=all_rubrics[index_zero],
                            category="zeroshot"
                        )
                        logger.info(f"LLMAESSScorer with rubrics {index_zero + 1}")
                    if "llmaes-scorer-fewshot" in model_spec_name:
                        index_few = model_spec_name.split("-")[-1]
                        index_few = int(index_few) - 1
                        output = llm_scorer.run_message(
                            model_name=request.model_llm,
                            set_essay=answer,
                            set_prompt=question,
                            set_rubrics=all_rubrics[index_few],
                            category="fewshot"
                        )
                        logger.info(f"LLMAESSScorer with fewshot rubrics {index_few + 1}")
                    time_seconds = time.time() - start_time
                    # content info
                    try:
                        content_info = output["content_score"]
                        results_out.append(f"LLMAESS-ContentScore")
                        factors.append(content_info["score"])
                        reasons.append(content_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing content info: {e}")
                        results_out.append("LLMAESS-ContentScore")
                        factors.append(-1.0)
                        reasons.append("Error processing content info")
                    # language info
                    try:
                        language_info = output["language_score"]
                        results_out.append("LLMAESS-LanguageScore")
                        factors.append(language_info["score"])
                        reasons.append(language_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing language info: {e}")
                        results_out.append("LLMAESS-LanguageScore")
                        factors.append(-1.0)
                        reasons.append("Error processing language info")
                    # structure info
                    try:
                        structure_info = output["structure_score"]
                        results_out.append("LLMAESS-StructureScore")
                        factors.append(structure_info["score"])
                        reasons.append(structure_info["reason"])
                    except Exception as e:
                        logger.error(f"Error processing structure info: {e}")
                        results_out.append("LLMAESS-StructureScore")
                        factors.append(-1.0)
                        reasons.append("Error processing structure info")
                    # overall info
                    try:
                        overall_info = output["total_score"]
                        results_out.append("LLMAESS-OverallScore")
                        factors.append(overall_info["score"])
                        reasons.append("Overall score based on all factors")
                    except Exception as e:
                        try:
                            # Try to compute overall score from individual scores
                            overall_score = output["content_score"]["score"] + output["language_score"]["score"] + output["structure_score"]["score"]
                            results_out.append("LLMAESS-OverallScore")
                            factors.append(overall_score)
                            reasons.append("Overall score computed from content, language, and structure scores")
                        except Exception as e:
                            logger.error(f"Error processing overall info: {e}")
                            results_out.append("LLMAESS-OverallScore")
                            factors.append(-1.0)
                            reasons.append("Error processing overall info")
                    for _ in range(4):
                        begin.append(request.answers[i]["begin"])
                        end.append(request.answers[i]["end"])
                        json_llm_string = json.dumps(output)
                        contents.append(output["output"])
                        responses.append(json_llm_string)
                        definitions.append("LLMAESS Score")
                        additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds, "model_spec_name": model_spec_name}))
                        answer_ids.append(request.answer_ids[i])
                        question_ids.append(request.question_ids[i])
                        nameLLMModel.append(request.name_model)
                        if len(all_scenes) > 0:
                            scene_ids.append(request.scenario_ids[i])
            case "AAGScorer":
                aag_scorer = AGGScore(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = aag_scorer.run_message(
                        model_name=request.model_llm,
                        course="Experiments",
                        question=question,
                        student_answer=answer
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    results_out.append("AAGScore")
                    factors.append(output["score"])
                    reasons.append(output["reason"])
                    json_llm_string = json.dumps(output)
                    contents.append(output["output"])
                    responses.append(json_llm_string)
                    definitions.append("AAG Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds, "model_spec_name": model_spec_name}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "GradingMedicalEducation":
                med_scorer = GradingMedicalEducationScorer(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = med_scorer.run_message(
                        model_name=request.model_llm,
                        course="",
                        question_text=question,
                        student_answer=answer,
                        key_text="<Score Point from 1 to 10>",
                        max_points=10,
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    results_out.append("MedScore")
                    factors.append(output["score"])
                    reasons.append(output["reason"])
                    json_llm_string = json.dumps(output)
                    contents.append(output["output"])
                    responses.append(json_llm_string)
                    definitions.append("Medical Education Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds, "model_spec_name": model_spec_name}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "AES-ASAP-MTS-Scorer":
                mts_scorer = MTS_OpenAI(
                    template_path="ASAP_template_mts.xlsx",
                    category="ASAP",
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    model_spec_name_split = settings.model_spec_name.split("-")
                    trait_id = model_spec_name_split[-1]
                    trait_id = int(trait_id)  # Convert to zero-based index
                    template_id = model_spec_name_split[-2]
                    template_id = int(template_id)  # Convert to zero-based index
                    output = mts_scorer.run_message(
                        template_id=template_id,
                        traid_id=trait_id,
                        task=question,
                        essay=answer,
                        model_name=request.model_llm
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    trait_name = output["score"]["trait"]
                    score_out = output["score"]["score"]
                    if score_out == -1:
                        reason_i = "Error processing score"
                    else:
                        reason_i = "Score based on the MTS model"
                    results_out.append(trait_name)
                    factors.append(score_out)
                    reasons.append(reason_i)
                    json_llm_string = json.dumps(output)
                    contents.append(output["score"]["output"])
                    responses.append(json_llm_string)
                    definitions.append("MTS Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "AES-ASAP-Vanilla-Scorer":
                vanilla_scorer = Vanilla_OpenAI(
                    template_path="ASAP_template_vanilla.xlsx",
                    category="ASAP",
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    model_spec_name_split = settings.model_spec_name.split("-")
                    template_id = model_spec_name_split[-1]
                    template_id = int(template_id) # Convert to zero-based index
                    output = vanilla_scorer.run_message(
                        template_id=template_id,
                        prompt=question,
                        essay=answer,
                        model_name=request.model_llm
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    trait_name = output["score"]["trait"]
                    score_out = output["score"]["score"]
                    if score_out == -1:
                        reason_i = "Error processing score"
                    else:
                        reason_i = "Score based on the Vanilla model"
                    results_out.append(trait_name)
                    factors.append(score_out)
                    reasons.append(reason_i)
                    json_llm_string = json.dumps(output)
                    contents.append(output["score"]["output"])
                    responses.append(json_llm_string)
                    definitions.append("Vanilla Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "AES-TOEFEL11-MTS-Scorer":
                mts_scorer = MTS_OpenAI(
                    template_path="toefel11_template_mts.xlsx",
                    category="TOEFEL11",
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    model_spec_name_split = settings.model_spec_name.split("-")
                    trait_id = model_spec_name_split[-1]
                    trait_id = int(trait_id)  # Convert to zero-based index
                    template_id = model_spec_name_split[-2]
                    template_id = int(template_id)  # Convert to zero-based index
                    output = mts_scorer.run_message(
                        template_id=template_id,
                        traid_id=trait_id,
                        task=question,
                        essay=answer,
                        model_name=request.model_llm
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    trait_name = output["score"]["trait"]
                    score_out = output["score"]["score"]
                    if score_out == -1:
                        reason_i = "Error processing score"
                    else:
                        reason_i = "Score based on the MTS model"
                    results_out.append(trait_name)
                    factors.append(score_out)
                    reasons.append(reason_i)
                    json_llm_string = json.dumps(output)
                    contents.append(output["score"]["output"])
                    responses.append(json_llm_string)
                    definitions.append("MTS Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "AES-TOEFEL11-Vanilla-Scorer":
                vanilla_scorer = Vanilla_OpenAI(
                    template_path="toefel11_template_vanilla.xlsx",
                    category="TOEFEL11",
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    model_spec_name_split = settings.model_spec_name.split("-")
                    template_id = model_spec_name_split[-1]
                    template_id = int(template_id) # Convert to zero-based index
                    output = vanilla_scorer.run_message(
                        template_id=template_id,
                        prompt=question,
                        essay=answer,
                        model_name=request.model_llm
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    trait_name = output["score"]["trait"]
                    score_out = output["score"]["score"]
                    if score_out == -1:
                        reason_i = "Error processing score"
                    else:
                        reason_i = "Score based on the Vanilla model"
                    results_out.append(trait_name)
                    factors.append(score_out)
                    reasons.append(reason_i)
                    json_llm_string = json.dumps(output)
                    contents.append(output["score"]["output"])
                    responses.append(json_llm_string)
                    definitions.append("Vanilla Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case "GradingLikeHumanScorer":
                human_scorer = GradingLikeHumanScorer(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = human_scorer.run_message(
                        model_name=request.model_llm,
                        question=question,
                        student_response=answer,
                        full_points=10
                    )
                    time_seconds = time.time() - start_time

                    results_out.append("GradingLikeHumanScoringScore")
                    factors.append(output["Extraction"]["Scoring"]["score"])
                    reasons.append("Score based on the Grade Like Human model")
                    results_out.append("GradingLikeHumanScoringConfidence")
                    factors.append(output["Extraction"]["Scoring"]["confidence"])
                    reasons.append("Confidence based on the Grade Like Human model")
                    results_out.append("GradingLikeHumanReflectionScore")
                    factors.append(output["Extraction"]["Reflection"]["score"])
                    reasons.append("Score based on the Grade Like Human reflection model")
                    results_out.append("GradingLikeHumanReflectionConfidence")
                    factors.append(output["Extraction"]["Reflection"]["confidence"])
                    reasons.append("Confidence based on the Grade Like Human reflection model")

                    for _ in range(4):
                        begin.append(request.answers[i]["begin"])
                        end.append(request.answers[i]["end"])
                        json_llm_string = json.dumps(output)
                        contents.append(output["Extraction"]["output"])
                        responses.append(json_llm_string)
                        definitions.append("Grading Like Human Score")
                        additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds, "model_spec_name": model_spec_name}))
                        answer_ids.append(request.answer_ids[i])
                        question_ids.append(request.question_ids[i])
                        nameLLMModel.append(request.name_model)
                        if len(all_scenes) > 0:
                            scene_ids.append(request.scenario_ids[i])
            case "EssayEvalScoring":
                essay_eval_scorer = EssayEvalScoring(
                    url=request.url,
                    port=request.port,
                    seed=request.seed,
                    temperature=request.temperature,
                    api_key=None
                )
                for i, (question, answer) in enumerate(zip(all_questions, all_answers)):
                    if len(all_scenes) > 0:
                        scene = all_scenes[i]
                        # scene_ids.append(request.scenario_ids[i])
                        question = f"Scene:{scene}\nTask:{question}"
                    start_time = time.time()
                    output = essay_eval_scorer.run_message(
                        model_name=request.model_llm,
                        task=question,
                        essay=answer
                    )
                    time_seconds = time.time() - start_time
                    begin.append(request.answers[i]["begin"])
                    end.append(request.answers[i]["end"])
                    results_out.append("EssayEvalScore")
                    factors.append(output["Scoring"]["score"])
                    reasons.append(output["Scoring"]["reason"])
                    json_llm_string = json.dumps(output)
                    contents.append(output["Scoring"]["output"])
                    responses.append(json_llm_string)
                    definitions.append("Essay Evaluation Score")
                    additional.append(json.dumps({"url": request.url, "port": request.port, "model_name": request.model_llm, "seed": request.seed, "temperature": request.temperature, "duration": time_seconds, "model_spec_name": model_spec_name}))
                    answer_ids.append(request.answer_ids[i])
                    question_ids.append(request.question_ids[i])
                    nameLLMModel.append(request.name_model)
                    if len(all_scenes) > 0:
                        scene_ids.append(request.scenario_ids[i])
            case _:
                raise ValueError(f"Model {model_name} is not supported.")
    output = {
        "begin": begin,
        "end": end,
        "keys": results_out,
        "values": factors,
        "definitions": definitions,
        "answer_ids": answer_ids,
        "question_ids": question_ids,
        "scene_ids": scene_ids,
        "contents": contents,
        "responses": responses,
        "additional": additional,
        "reasons": reasons,
        "nameLLMModel": nameLLMModel,
    }
    return output



# Response sent by DUUI
# Note, this is transformed by the Lua script
class DUUIResponse(BaseModel):
    # Symspelloutput
    # List of Sentence with every token
    # Every token is a dictionary with following Infos:
    # Symspelloutput right if the token is correct, wrong if the token is incorrect, skipped if the token was skipped, unkownn if token can corrected with Symspell
    # If token is unkown it will be predicted with BERT Three output pos:
    # 1. Best Prediction with BERT MASKED
    # 2. Best Cos-sim with Sentence-Bert and with perdicted words of BERT MASK
    # 3. Option 1 and 2 together
    meta: AnnotationMeta
    # Modification meta, one per document
    modification_meta: DocumentModification
    begin: List[int]
    end: List[int]
    values: List
    keys: List
    definitions: List
    answer_ids: List[str]
    question_ids: List[str]
    scene_ids: List[str]
    model_name: str
    model_version: str
    model_source: str
    model_lang: str
    contents: List[str]
    responses: List[str]
    additional: List[str]
    reasons: List[str]
    llmUsed: str
    NameModel: List[str] # List of model names used for scoring, if applicable


app = FastAPI(
    openapi_url="/openapi.json",
    docs_url="/api",
    redoc_url=None,
    title=settings.annotator_name,
    description="EssayScorer",
    # version=settings.model_version,
    terms_of_service="https://www.texttechnologylab.org/legal_notice/",
    contact={
        "name": "TTLab Team",
        "url": "https://texttechnologylab.org",
        "email": "bagci@em.uni-frankfurt.de",
    },
    license_info={
        "name": "AGPL",
        "url": "http://www.gnu.org/licenses/agpl-3.0.en.html",
    },
)

with open(lua_communication_script_filename, 'rb') as f:
    lua_communication_script = f.read().decode("utf-8")
logger.debug("Lua communication script:")
logger.debug(lua_communication_script_filename)


# Get typesystem of this annotator
@app.get("/v1/typesystem")
def get_typesystem() -> Response:
    # TODO remove cassis dependency, as only needed for typesystem at the moment?
    xml = typesystem.to_xml()
    xml_content = xml.encode("utf-8")

    return Response(
        content=xml_content,
        media_type="application/xml"
    )


# Return Lua communication script
@app.get("/v1/communication_layer", response_class=PlainTextResponse)
def get_communication_layer() -> str:
    return lua_communication_script


# Return documentation info
@app.get("/v1/documentation")
def get_documentation():
    return f"EssayScorer: {settings.model_name}"


# Process request from DUUI
@app.post("/v1/process")
def post_process(request: DUUIRequest):
    # Return data
    # Save modification start time for later
    modification_timestamp_seconds = int(time.time())
    begin = []
    end = []
    values = []
    keys = []
    definitions = []
    answer_ids = []
    question_ids = []
    scene_ids = []
    contents = []
    responses = []
    additional = []
    reasons = []
    nameLLMModel = []
    llm_list = {"BeGradingScorer", "LLMAESSlowScorer", "LLMAESSScorer", "AAGScorer", "GradingMedicalEducation", "GradingLikeHumanScorer", "EssayEvalScoring"}
    llm_used = "No"
    try:
        if settings.model_name in llm_list:
            llm_used = "Yes"
        # set meta Informations
        meta = AnnotationMeta(
            name=settings.annotator_name,
            version=settings.annotator_version,
            modelName=settings.model_name,
            modelVersion=settings.model_version,
        )
        # Add modification info
        modification_meta_comment = f"{settings.annotator_name} ({settings.annotator_version}))"
        modification_meta = DocumentModification(
            user=settings.annotator_name,
            timestamp=modification_timestamp_seconds,
            comment=modification_meta_comment
        )
        processed_sentences = process_selection(request, settings.model_name)
        begin.extend(processed_sentences["begin"])
        end.extend(processed_sentences["end"])
        values.extend(processed_sentences["values"])
        keys.extend(processed_sentences["keys"])
        definitions.extend(processed_sentences["definitions"])
        answer_ids.extend(processed_sentences["answer_ids"])
        question_ids.extend(processed_sentences["question_ids"])
        scene_ids.extend(processed_sentences["scene_ids"])
        contents.extend(processed_sentences["contents"])
        responses.extend(processed_sentences["responses"])
        additional.extend(processed_sentences["additional"])
        reasons.extend(processed_sentences["reasons"])
        nameLLMModel.extend(processed_sentences["nameLLMModel"])
    except Exception as ex:
        logger.exception(ex)
    return DUUIResponse(meta=meta, modification_meta=modification_meta, begin=begin, end=end, values=values, keys=keys, definitions=definitions, answer_ids=answer_ids, question_ids=question_ids, scene_ids=scene_ids, model_source=settings.model_source, model_name=settings.model_name, model_version=settings.model_version, model_lang=settings.model_lang, contents=contents, responses=responses, additional=additional, reasons=reasons,llmUsed=llm_used, NameModel=nameLLMModel)

@lru_cache_with_size
def load_model(model_name: str, device: str) -> EssayScorer:
    """
    Load the model with caching.
    :param model_name: Name of the model to load.
    :param device: Device to load the model on (e.g., 'cpu' or 'cuda').
    :return: Loaded EssayScorer model.
    """
    if model_name in {"KevSun/Engessay_grading_ML", "JacobLinCool/IELTS_essay_scoring_safetensors"}:
        scorer = EssayScorer(model_name=model_name, device=device)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return scorer
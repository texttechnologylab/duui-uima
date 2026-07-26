StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Divs = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Div")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    -- multiple selection types can be given, separated by commas
    local div_questions = parameters["div_questions"]
    local div_answers = parameters["div_answers"]
    local div_scenarios = parameters["div_scenarios"]

    local seed = parameters["seed"]
    local temperature = parameters["temperature"]
    local url = parameters["url"]
    local port = parameters["port"]
    local model_llm = parameters["model_llm"]
    local name_model = parameters["name_model"]
    if seed == nil then
        seed = -100
    end
    if temperature == nil then
        temperature = -100.0
    end
    if url == nil then
        url = ""
    end
    if port == nil then
        port = -100
    end
    if model_llm == nil then
        model_llm = ""
    end

    if name_model == nil then
        name_model = "No model name given"
    end


    div_id_map = {}
    local all_divs = JCasUtil:select(inputCas, Divs):iterator()
    while all_divs:hasNext() do
        local div = all_divs:next()
--         print(div)
        local div_name = div:getType():getName()
        if div_name == "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Div" then
            local div_id = div:getId()
--             print(div_id)
            div_id_map[div_id] = div
        end
    end

    answers = {}
    questions = {}
    scenarios = {}
    answer_id = {}
    question_id = {}
    scenario_id = {}

    print(div_questions)
    i = 1
    if div_questions ~= nil then
        for div_id in string.gmatch(div_questions, '([^,]+)') do
            local div = div_id_map[div_id]
            if div ~= nil then
                print(div_id)
                local question = {
                    text = div:getCoveredText(),
                    begin = div:getBegin(),
                    ['end'] = div:getEnd(),
                    typeName = div_id
                }
                questions[i] = question
                question_id[i] = div_id
                i = i + 1
            end
        end
    end

    print(div_answers)
    i = 1
    if div_answers ~= nil then
--         print(div_answers)
        for div_id in string.gmatch(div_answers, '([^,]+)') do
--             print(div_id)
            local div = div_id_map[div_id]
            if div ~= nil then
                print(div_id)
                local answer = {
                    text = div:getCoveredText(),
                    begin = div:getBegin(),
                    ['end'] = div:getEnd(),
                    typeName = div_id
                }
                answers[i] = answer
                answer_id[i] = div_id
                i = i + 1
            end
        end
    end

    print(div_scenarios)
    i = 1
    if div_scenarios ~= nil then
        for div_id in string.gmatch(div_scenarios, '([^,]+)') do
            local div = div_id_map[div_id]
            if div ~= nil then
                print(div_id)
                local scenario = {
                    text = div:getCoveredText(),
                    begin = div:getBegin(),
                    ['end'] = div:getEnd(),
                    typeName = div_id
                }
                scenarios[i] = scenario
                scenario_id[i] = div_id
                i = i + 1
            end
        end
    end



    outputStream:write(json.encode({
        lang = doc_lang,
        doc_len = doc_len,
        questions = questions,
        question_ids = question_id,
        answers = answers,
        answer_ids = answer_id,
        scenarios = scenarios,
        scenario_ids = scenario_id,
        seed = seed,
        temperature = temperature,
        url = url,
        port = port,
        model_llm = model_llm,
        name_model = name_model,
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
     local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
     local results = json.decode(inputString)
     if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["values"] ~= nil then
 --         print("GetInfo")
         local source = results["model_source"]
         local model_version = results["model_version"]
         local model_name = results["model_name"]
         local model_lang = results["model_lang"]
 --         print("meta")
         local modification_meta = results["modification_meta"]
         local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
         modification_anno:setUser(modification_meta["user"])
         modification_anno:setTimestamp(modification_meta["timestamp"])
         modification_anno:setComment(modification_meta["comment"])
         modification_anno:addToIndexes()

 --         print("setMetaData")
         local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
         model_meta:setModelVersion(model_version)
--          print(model_version)
         model_meta:setModelName(model_name)
--          print(model_name)
         model_meta:setSource(source)
--          print(source)
         model_meta:setLang(model_lang)
--          print(model_lang)
         model_meta:addToIndexes()

         local meta = results["meta"]
         local begin_read = results["begin"]
         local end_read = results["end"]
         local values = results["values"]
         local keys = results["keys"]
         local definitions = results["definitions"]
         local answer_ids = results["answer_ids"]
         local question_ids = results["question_ids"]
         local scenario_ids = results["scene_ids"]
         local contents = results["contents"]
         local responses = results["responses"]
         local additional = results["additional"]
         local reasons = results["reasons"]
         local llmUsed = results["llmUsed"]
         local nameLLMModel = results["NameModel"]
--          print(nameLLMModel)
--          print("LLMUsed: " .. llmUsed)


         for i, value in ipairs(values) do
             local begin_i = begin_read[i]
             local end_i = end_read[i]
             local value_i = values[i]
             local key_i = keys[i]
             local def_i = definitions[i]
             local answer_id_i =answer_ids[i]
             local question_id_i = nil
             if question_ids ~= nil then
                 question_id_i = question_ids[i]
             end
             local scenario_id_i = nil
             if scenario_ids ~= nil then
                 scenario_id_i = scenario_ids[i]
             end
             local content_i = nil
             if contents ~= nil then
                 content_i = contents[i]
             end
             local response_i = nil
             if responses ~= nil then
                 response_i = responses[i]
             end
             local additional_i = nil
             if additional ~= nil then
                 additional_i = additional[i]
             end
             local reason_i = nil
             if reasons ~= nil then
                 reason_i = reasons[i]
             end
             local name_llm_model_i = nil
             if nameLLMModel ~= nil then
                 name_llm_model_i = nameLLMModel[i]
--                  print(name_llm_model_i)
             end
--              print("step1")

             local essay_score = luajava.newInstance("org.texttechnologylab.annotation.EssayScore", inputCas)
             essay_score:setBegin(begin_i)
             essay_score:setEnd(end_i)
             essay_score:setValue(value_i)
             essay_score:setName(key_i)
             if llmUsed == "Yes" then
                 if reason_i ~= nil then
                     essay_score:setReason(reason_i)
--                      print(reason_i)
                 else
                     essay_score:setReason("LLM used, no reason given")
                 end
             else
                essay_score:setReason("")
             end
--              print("step2")

             local comment_answer = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
             comment_answer:setReference(essay_score)
             comment_answer:setKey("div_answers")
             comment_answer:setValue(answer_id_i)
             comment_answer:addToIndexes()
             essay_score:setInputAnswer(comment_answer)
             if question_id_i ~= nil then
                 local comment_question = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                 comment_question:setReference(essay_score)
                 comment_question:setKey("div_questions")
                 comment_question:setValue(question_id_i)
                 comment_question:addToIndexes()
                 essay_score:setInputQuestion(comment_question)
             end
--              print("step3")

             if scenario_id_i ~= nil then
                 local comment_scenario = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                 comment_scenario:setReference(essay_score)
                 comment_scenario:setKey("div_scenarios")
                 comment_scenario:setValue(scenario_id_i)
                 comment_scenario:addToIndexes()
                 essay_score:setInputScene(comment_scenario)
             end
             essay_score:addToIndexes()
--              print("step4")
             local EssayScoreModel = nil
            -- Model Def
             if llmUsed == "Yes" then
                 print("LLM used")
                EssayScoreModel = luajava.newInstance("org.texttechnologylab.annotation.model.EssayScoreLLM", inputCas)
                if content_i ~= nil then
                    EssayScoreModel:setContents(content_i)
--                     print(content_i)
                end
                if response_i ~= nil then
                    EssayScoreModel:setResponse(response_i)
--                     print(response_i)
                end
                if additional_i ~= nil then
                    EssayScoreModel:setAdditionalInformation(additional_i)
--                     print(additional_i)
                end
                if name_llm_model_i ~= nil then
                    EssayScoreModel:setModelName(name_llm_model_i)
--                     print(name_llm_model_i)
                end
             else
                print("LLM not used")
                EssayScoreModel = luajava.newInstance("org.texttechnologylab.annotation.model.EssayScoreModel", inputCas)
             end
--              print("step5")

             EssayScoreModel:setBegin(begin_i)
             EssayScoreModel:setEnd(end_i)
             EssayScoreModel:setModel(model_meta)
             EssayScoreModel:setScoreReference(essay_score)
             EssayScoreModel:addToIndexes()
         end
     end
 end

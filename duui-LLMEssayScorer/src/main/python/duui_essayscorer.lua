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

    div_id_map = {}
    local all_divs = JCasUtil:select(inputCas, Divs):iterator()
    while all_divs:hasNext() do
        local div = all_divs:next()
        print(div)
        local div_id = div:getId()
        div_id_map[div_id] = div
    end

    answers = {}
    questions = {}

    print(div_questions)
    i = 1
    if div_questions ~= nil then
        for div_id in string.gmatch(div_questions, '([^,]+)') do
            local div = div_id_map[div_id]
            if div ~= nil then
                local question = {
                    text = div:getCoveredText(),
                    begin = div:getBegin(),
                    ['end'] = div:getEnd(),
                    typeName = div_id
                }
                questions[i] = question
            end
            i = i + 1
        end
    end

    print(div_answers)
    i = 1
    if div_answers ~= nil then
        print(div_answers)
        for div_id in string.gmatch(div_answers, '([^,]+)') do
            print(div_id)
            local div = div_id_map[div_id]
            if div ~= nil then
                local answer = {
                    text = div:getCoveredText(),
                    begin = div:getBegin(),
                    ['end'] = div:getEnd(),
                    typeName = div_id
                }
                answers[i] = answer
            end
            i = i + 1
        end
    end



    outputStream:write(json.encode({
        lang = doc_lang,
        doc_len = doc_len,
        questions = questions,
        answers = answers
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
         local len_results = results["len_results"]


         for i, value in ipairs(values) do
             local begin_i = begin_read[i]
--              print(begin_i)
             local end_i = end_read[i]
             local len_i = len_results[i]

             local value_i = values[i]
             local key_i = keys[i]
--              print(end_i)
             local def_i = definitions[i]
--              print(def_i)

             for j, key_j in ipairs(key_i) do
--                  print(j)
                 local llm_detect = luajava.newInstance("org.texttechnologylab.annotation.LLMMetric", inputCas)
                 value_j = value_i[j]
                 key_j = key_i[j]
--                  print(key_j)
                 def_j = def_i[j]
                 llm_detect:setBegin(begin_i)
                 llm_detect:setEnd(end_i)
                 llm_detect:setValue(value_j)
                 llm_detect:setKeyName(key_j)
                 llm_detect:setDefinition(def_j)
                 llm_detect:setModel(model_meta)
                 llm_detect:addToIndexes()
             end
         end
     end
 end

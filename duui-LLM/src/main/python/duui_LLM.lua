StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Prompts = luajava.bindClass("org.texttechnologylab.type.LLMPrompt")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
--     print(model_name)
    local seed = parameters["seed"]
    local temperature = parameters["temperature"]
    local url = parameters["url"]
    local port = parameters["port"]
    local model_name = parameters["model_name"]
--     print(model_name)

    local all_prompts = {}
--     print(select)

    local selections = {}
    local selections_count = 1
    local prompt_counter = 0
    local prompts_in = util:select(inputCas, Prompts):iterator()
    print(prompts_in)
    while prompts_in:hasNext() do
        local prompt = prompts_in:next()
        local begin_prompt = prompt:getBegin()
--         print(begin_prompt)
        local end_prompt = prompt:getEnd()
--         print(end_prompt)
        local prompt_text = prompt:getPrompt()
--         print(prompt_text)
        local prefix = prompt:getPrefix()
--         print(prefix)
        local suffix = prompt:getSuffix()
--         print(suffix)
        local systemPrompt = prompt:getSystemPrompt()
--         print(systemPrompt)
        -- check if prefix, suffix and systemPrompt are not nil
        local prefix_in
        if prefix == nil then
            prefix_in = {
                text = "",
                begin = 0,
                ['end'] = 0
            }
        else
            prefix_in = {
                text = prefix:getMessage(),
                begin = prefix:getBegin(),
                ['end'] = prefix:getEnd()
            }
        end
        local suffix_in
        if suffix == nil then
            suffix_in = {
                text = "",
                begin = 0,
                ['end'] = 0
            }
        else
            suffix_in = {
                text = suffix:getMessage(),
                begin = suffix:getBegin(),
                ['end'] = suffix:getEnd()
            }
        end
        local systemPrompt_in
        if systemPrompt == nil then
            systemPrompt_in = {
                text = "",
                begin = 0,
                ['end'] = 0
            }
        else
            systemPrompt_in = {
                text = systemPrompt:getMessage(),
                begin = systemPrompt:getBegin(),
                ['end'] = systemPrompt:getEnd()
            }
        end
        all_prompts[selections_count] = {
            text = prompt_text,
            begin = begin_prompt,
            prefix = prefix_in,
            suffix = suffix_in,
            id = prompt_counter,
            systemPrompt = systemPrompt_in,
            ['end'] = end_prompt
        }
        prompt_counter = prompt_counter + 1
        selections_count = selections_count + 1
    end
--     print(all_prompts)
    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        len = doc_len,
        prompts = all_prompts,
        seed = seed,
        temperature = temperature,
        url = url,
        port = port,
        model_name = model_name
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["responses"] ~= nill and results["meta"] ~=nil and results["modification_meta"]~= nil then
--         print("GetInfo")
        local  modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]
        local begin_prompts=results["begin_prompts"]
        local end_prompts=results["end_prompts"]
        local id_prompts=results["id_prompts"]
        local responses=results["responses"]
        local contents=results["contents"]
        local additional=results["additional"]

        for index_i, begin_prompt in ipairs(begin_prompts) do
            local begin_prompt_i = begin_prompt
--             print("begin_prompt_i")
            local end_prompt_i = end_prompts[index_i]
--             print(end_prompt_i)
            local id_prompt_i = id_prompts[index_i]
            print(id_prompt_i)
--             print("end_prompt_i")
            local response_i = responses[index_i]
--             print("response_i")
            local content_i = contents[index_i]
--             print("content_i")
            local additional_i = additional[index_i]
--             print("additional_i")

            local prompt_i = util:selectByIndex(inputCas, Prompts, id_prompt_i)
            print(prompt_i)
--             print("prompt_i")

            local LLMResult = luajava.newInstance("org.texttechnologylab.type.LLMResult", inputCas)
--             print("LLmResult")
            LLMResult:setBegin(begin_prompt_i)
--             print("setBegin")
            LLMResult:setEnd(end_prompt_i)
--             print("setEnd")
            LLMResult:setMeta(additional_i)
--             print("setMeta")
            LLMResult:setResult(response_i)
--             print("setResult")
            LLMResult:setContent(content_i)
            LLMResult:setPrompt(prompt_i)
--             print("setContent")
            LLMResult:addToIndexes()
--             print("addToIndexes")
        end
    end
end

StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Prompt = luajava.bindClass("org.texttechnologylab.type.llm.prompt.Prompt")

function serialize(inputCas, outputStream, parameters)
    local llm_args = parameters["llm_args"]
    if llm_args == nil then
        llm_args = "{}"
    end

    local prompts = {}
    local prompts_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Prompt)):listIterator()
    local prompt_count = 1
    while prompts_it:hasNext() do
        local prompt = prompts_it:next()

        local messages = {}
        local messages_it = prompt:getMessages():iterator()
        local messages_count = 1
        while messages_it:hasNext() do
            local message = messages_it:next()
            messages[messages_count] = {
                role = message:getRole(),
                content = message:getContent()
            }
            messages_count = messages_count + 1
        end

        prompts[prompt_count] = {
            args = prompt:getArgs(),
            messages = messages,
            ref = prompt:getAddress()
        }
        prompt_count = prompt_count + 1

    end

    outputStream:write(json.encode({
        prompts = prompts,
        llm_args = llm_args
    }))
end

function deserialize(inputCas, inputStream)
--     local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
--     local results = json.decode(inputString)
--
--     if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["sentences"] ~= nil then
--         local modification_meta = results["modification_meta"]
--         local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
--         modification_anno:setUser(modification_meta["user"])
--         modification_anno:setTimestamp(modification_meta["timestamp"])
--         modification_anno:setComment(modification_meta["comment"])
--         modification_anno:addToIndexes()
--
--         local meta = results["meta"]
--         for j, sentence in ipairs(results["sentences"]) do
--             local sent_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", inputCas)
--             sent_anno:setBegin(sentence["begin"])
--             sent_anno:setEnd(sentence["end"])
--             sent_anno:addToIndexes()
--
--             local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
--             meta_anno:setReference(sent_anno)
--             meta_anno:setName(meta["name"])
--             meta_anno:setVersion(meta["version"])
--             meta_anno:setModelName(meta["modelName"])
--             meta_anno:setModelVersion(meta["modelVersion"])
--             meta_anno:addToIndexes()
--         end
--     end
end

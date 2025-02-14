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
                content = message:getContent(),
                ref = message:getAddress()
            }

            -- the message might contain module/class info
            if message:getClassModule() ~= nil then
                messages[messages_count]["class_module"] = message:getClassModule()
            end
            if message:getClassName() ~= nil then
                messages[messages_count]["class_name"] = message:getClassName()
            end

            -- check if "fillable" message
            if  message:getType():getName() == "org.texttechnologylab.type.llm.prompt.FillableMessage" then
                messages[messages_count]["fillable"] = true

                -- check if "context_name" is set, only on fillable messages
                if message:getContextName() ~= nil then
                    messages[messages_count]["context_name"] = message:getContextName()
                end
            end

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
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["llm_results"] ~= nil and results["prompts"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        -- "fill" the messages of the prompts
        for i, prompt in ipairs(results["prompts"]) do
            for j, message in pairs(prompt["messages"]) do
                if message["fillable"] == true then
                    local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(message["ref"])
                    msg_anno:setContent(message["content"])
                end
            end
        end

        local meta = results["meta"]
        for i, llm_result in ipairs(results["llm_results"]) do
            local llm_anno = luajava.newInstance("org.texttechnologylab.type.llm.prompt.Result", inputCas)
            llm_anno:setMeta(llm_result["meta"])
            local prompt_anno = inputCas:getLowLevelCas():ll_getFSForRef(llm_result["prompt_ref"])
            llm_anno:setPrompt(prompt_anno)
            local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(llm_result["message_ref"])
            llm_anno:setMessage(msg_anno)
            llm_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(llm_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

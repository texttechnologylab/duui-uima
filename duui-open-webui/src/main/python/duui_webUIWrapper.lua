StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Prompt = luajava.bindClass("org.texttechnologylab.type.llm.prompt.Prompt")
Image = luajava.bindClass("org.texttechnologylab.annotation.type.Image")
Audio = luajava.bindClass("org.texttechnologylab.annotation.type.Audio")

function serialize(inputCas, outputStream, parameters)
    print("start serilize")

    local doc_lang = inputCas:getDocumentLanguage()

    -- Get parameters or use defaults
    local model_name = parameters["model_name"] or "llama3"
    local individual = parameters["individual"] or "false"
    local mode = parameters["mode"] or "text"
    local ollama_host = parameters["ollama_host"] or "http://localhost"
    local ollama_port = parameters["ollama_port"] or ""
    local ollama_auth_token = parameters["ollama_auth_token"] or ""

    local system_prompt = parameters["system_prompt"] or ""

    -- Prompts handler
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
            messages_count = messages_count + 1
        end
        prompts[prompt_count] = {
            args = prompt:getArgs(),
            messages = messages,
            ref = prompt:getAddress()
        }
        prompt_count = prompt_count + 1
    end
    print("start Image loader")

    -- Images handler
    local images = {}
    local number_of_images = 1
    local image_it = JCasUtil:select(inputCas, Image):iterator()
    while image_it:hasNext() do
        local image = image_it:next()
        images[number_of_images] = {
            src = image:getSrc(),
            height = image:getHeight(),
            width = image:getWidth(),
            begin = image:getBegin(),
            ['end'] = image:getEnd()
        }
        number_of_images = number_of_images + 1
    end

    -- Audios handler
    local audios = {}
    local number_of_audios = 1
    local audio_it = JCasUtil:select(inputCas, Audio):iterator()
    while audio_it:hasNext() do
        local audio = audio_it:next()
        audios[number_of_audios] = {
            src = audio:getSrc(),
            begin = audio:getBegin(),
            ['end'] = audio:getEnd()
        }
        number_of_audios = number_of_audios + 1
    end

    -- Videos handler
    local videos = {}
    local number_of_videos = 1
    local class = Class:forName("org.texttechnologylab.annotation.type.Video")
    local video_it = JCasUtil:select(inputCas, class):iterator()
    while video_it:hasNext() do
        local video = video_it:next()
        videos[number_of_videos] = {
            src = video:getSrc(),
            length = video:getLength(),
            fps = video:getFps(),
            begin = video:getBegin(),
            ['end'] = video:getEnd()
        }
        number_of_videos = number_of_videos + 1
    end

    -- Serialize to JSON
    outputStream:write(json.encode({
        images = images,
        audios = audios,
        videos = videos,
        prompts = prompts,
        doc_lang = doc_lang,
        model_name = model_name,
        individual = individual,
        mode = mode,
        ollama_host = ollama_host,
        ollama_port = ollama_port,
        ollama_auth_token = ollama_auth_token,
        system_prompt = system_prompt
    }))
end

-- function deserialize(inputCas, inputStream)
--     local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
--     local results = json.decode(inputString)
--
--     -- Handle errors
--     if results['errors'] ~= nil then
--         for _, error in ipairs(results['errors']) do
--             local warning = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
--             warning:setKey("error")
--             warning:setValue(error['meta'] or error)
--             warning:addToIndexes()
--         end
--     end
--
--     -- Handle model metadata
--     if results['model_source'] ~= nil and results['model_version'] ~= nil and results['model_name'] ~= nil and results['model_lang'] ~= nil then
--         local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
--         model_meta:setModelVersion(results["model_version"])
--         model_meta:setModelName(results["model_name"])
--         model_meta:setSource(results["model_source"])
--         model_meta:setLang(results["model_lang"])
--         model_meta:addToIndexes()
--     end
--
--     -- Handle prompts
--     if results['prompts'] ~= nil then
--         for _, prompt in ipairs(results["prompts"]) do
--             for _, message in pairs(prompt["messages"]) do
--                 if message["fillable"] == true then
--                     local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(message["ref"])
--                     msg_anno:setContent(message["content"])
--                 end
--             end
--         end
--     end
--
--     -- Handle processed text
--     if results['processed_text'] ~= nil then
--         for _, llm_result in ipairs(results["processed_text"]) do
--             local llm_anno = luajava.newInstance("org.texttechnologylab.type.llm.prompt.Result", inputCas)
--             llm_anno:setMeta(llm_result["meta"])
--             local prompt_anno = inputCas:getLowLevelCas():ll_getFSForRef(llm_result["prompt_ref"])
--             llm_anno:setPrompt(prompt_anno)
--             llm_anno:addToIndexes()
--         end
--     end
-- end

function deserialize(inputCas, inputStream)
    -- 1. Parse Input
    local StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    -- 2. Handle Errors (Existing Logic)
    if results['errors'] ~= nil then
        for _, error in ipairs(results['errors']) do
            local warning = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning:setKey("error")
            warning:setValue(error['meta'] or error)
            warning:addToIndexes()
        end
    end

    -- 3. Handle Model Metadata (Existing Logic)
    if results['model_source'] ~= nil then
        local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(results["model_version"])
        model_meta:setModelName(results["model_name"])
        model_meta:setSource(results["model_source"])
        model_meta:setLang(results["model_lang"])
        model_meta:addToIndexes()
    end

    -- 4. Handle Prompts (Existing Logic)
    if results['prompts'] ~= nil then
        for _, prompt in ipairs(results["prompts"]) do
            for _, message in pairs(prompt["messages"]) do
                if message["fillable"] == true then
                    local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(message["ref"])
                    msg_anno:setContent(message["content"])
                end
            end
        end
    end

    -- 5. NEW: Build Sofa String & Calculate Offsets
    -- We use Java's StringBuilder to ensure offsets match UIMA's Java character counting
    local sb = luajava.newInstance("java.lang.StringBuilder")
    local pending_results = {}
    local pending_prompts = {}

    -- 1. Process Prompts into StringBuilder
    if results['prompts'] ~= nil then
        for _, prompt_data in ipairs(results["prompts"]) do
            for _, message in pairs(prompt_data["messages"]) do
                local text = message["content"] or ""

                local b = sb:length()
                sb:append("User: "):append(text):append("\n\n")
                local e = sb:length()

                -- Update the existing Prompt/Message if it was passed by reference
                if message["fillable"] == true and message["ref"] ~= nil then
                    local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(message["ref"])
                    msg_anno:setBegin(b)
                    msg_anno:setEnd(e)
                    msg_anno:setContent(text)
                end
            end
        end
    end

    -- 2. Process Results into StringBuilder
    if results['processed_text'] ~= nil then
        for _, llm_result in ipairs(results["processed_text"]) do
            local content = llm_result["meta"] or ""

            local b = sb:length()
            sb:append("Assistant: "):append(content):append("\n\n")
            local e = sb:length()

            table.insert(pending_results, {
                begin_idx = b,
                end_idx = e,
                meta = content,
                prompt_ref = llm_result["prompt_ref"]
            })
        end
    end

    -- 3. Safely set Document Text
    local currentText = inputCas:getDocumentText()
    if currentText == nil then
        inputCas:setDocumentText(sb:toString())

        -- 4. Only add Result annotations if we successfully set the text
        for _, item in ipairs(pending_results) do
            local llm_anno = luajava.newInstance("org.texttechnologylab.type.llm.prompt.Result", inputCas)
            llm_anno:setBegin(item.begin_idx)
            llm_anno:setEnd(item.end_idx)
            llm_anno:setMeta(item.meta)

            if item.prompt_ref then
                local prompt_anno = inputCas:getLowLevelCas():ll_getFSForRef(item.prompt_ref)
                llm_anno:setPrompt(prompt_anno)
            end
            llm_anno:addToIndexes()
        end
    else
        print("Sofa already set. Skipping annotation creation to prevent offset mismatch.")
    end
end

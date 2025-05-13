
--package.path = package.path .. ";/home/staff_homes/aabusale/localgit/MobDebug/src/?.lua"
--local mobdebug = require("mobdebug")
--mobdebug.start()


StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Prompt = luajava.bindClass("org.texttechnologylab.type.llm.prompt.Prompt")
--Video = luaJava.bindClass("org.texttechnologylab.annotation.type.Video")

Image = luajava.bindClass("org.texttechnologylab.annotation.type.Image")
Audio = luajava.bindClass("org.texttechnologylab.annotation.type.Audio")

function serialize(inputCas, outputStream, parameters)
    print("start Lua serialzation")
    local doc_lang = inputCas:getDocumentLanguage()

    --print("doc_lang: ", doc_lang)

    -- get the parameters or default
    local model_name = parameters["model_name"] if parameters["model_name"] == nil then model_name = "microsoft/Phi-4-multimodal-instruct" end
    local selection_types = parameters["selections"] if parameters["selections"] == nil then selection_types="org.texttechnologylab.annotation.type.Image" end
    local individual = parameters["individual"] if parameters["individual"] == nil then individual = "false" end
    local mode = parameters['mode'] if parameters['mode'] == nil then mode = 'text'

    end
    --print("truncate_text: ", truncate_text)
    --print("start")
    --local images = {}
    --local number_of_images = 1
    --for selection_type in string.gmatch(selection_types, "([^,]+)") do
    --    print("selection_type: ", selection_type)
    --    if selection_type == 'text' then
    --        local doc_text = inputCas:getDocumentText()
    --        local doc_len = TopicUtils:getDocumentTextLength(inputCas)
    --        images[number_of_images] = {
    --            src = doc_text,
    --            height = 0,
    --            width = 0,
    --            begin = 0,
    --            ['end'] = doc_len
    --        }
    --        number_of_images = number_of_images + 1
    --    else
    --        local class = Class:forName(selection_type);
    --        local image_it = JCasUtil:select(inputCas, class):iterator()
    --        while image_it:hasNext() do
    --            local image = image_it:next()
    --            images[number_of_images] = {
    --                src = image:getSrc(),
    --                height = image:getHeight(),
    --                width = image:getWidth(),
    --                begin = image:getBegin(),
    --                ['end'] = image:getEnd()
    --            }
    --            number_of_images = number_of_images + 1
    --        end
    --
    --    end
    --
    --end


    -- prompts handler
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

        end
        prompts[prompt_count] = {
            args = prompt:getArgs(),
            messages = messages,
            ref = prompt:getAddress()
        }
        prompt_count = prompt_count + 1

    end

    print("start image")
    -- images handler if exists
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


    print("start audio")

    -- TODO: add audio handler
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

    print("start video selection")
    -- TODO: Add Video Support
    local videos = {}
    --local number_of_videos = 1
    --local video_it = JCasUtil:select(inputCas, Video):iterator()
    --while video_it:hasNext() do
    --    local video = video_it:next()
    --    videos[number_of_videos] = {
    --        src = video:getSrc(),
    --        length = video:getLength(),
    --        fps = video:getFps(),
    --        begin = video:getBegin(),
    --        ['end'] = video:getEnd()
    --    }
    --    number_of_videos = number_of_videos + 1
    --end


    outputStream:write(json.encode({
        images = images,
        audios = audios,
        videos = videos,
        prompts = prompts,
        doc_lang = doc_lang,
        model_name = model_name,
        individual = individual,
        mode = mode
    }))
end

function deserialize(inputCas, inputStream)
    print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    --print("results")
    --print(results)

    if results['errors'] ~= nil then
        local errors = results['errors']
        for index_i, error in ipairs(errors) do
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning_i:setKey("error")
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end
    --print("---------------------- Finished errors ----------------------")

    if results['model_source'] ~= nil and results['model_version'] ~= nil and results['model_name'] ~= nil and results['model_lang'] ~= nil then
        --print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

        --print("setMetaData")
        local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(model_version)
        --         print(model_version)
        model_meta:setModelName(model_name)
        --         print(model_name)
        model_meta:setSource(source)
        --         print(source)
        model_meta:setLang(model_lang)
        --         print(model_lang)
        model_meta:addToIndexes()

    end

    if results['images'] ~= nil then
        -- add images
    end


    if results['prompts'] ~= nil then
        for i, prompt in ipairs(results["prompts"]) do
            for j, message in pairs(prompt["messages"]) do
                if message["fillable"] == true then
                    local msg_anno = inputCas:getLowLevelCas():ll_getFSForRef(message["ref"])
                    msg_anno:setContent(message["content"])
                end
            end
        end

    end

    if results['processed_text'] ~= nil then
        for i, llm_result in ipairs(results["processed_text"]) do
            local llm_anno = luajava.newInstance("org.texttechnologylab.type.llm.prompt.Result", inputCas)
            llm_anno:setMeta(llm_result["meta"])
            local prompt_anno = inputCas:getLowLevelCas():ll_getFSForRef(llm_result["prompt_ref"])
            llm_anno:setPrompt(prompt_anno)
            llm_anno:addToIndexes()
        end

    end
    -- copy cas into a new one with new document text
end

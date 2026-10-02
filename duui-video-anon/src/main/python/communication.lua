local StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
local Class = luajava.bindClass("java.lang.Class")
local JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
local Video = Class:forName("org.texttechnologylab.annotation.type.Video")

local function parameter(params, key)
    if params == nil then return nil end
    local value = params[key]
    if value ~= nil then return tostring(value) end
    local ok, result = pcall(function() return params:get(key) end)
    if ok and result ~= nil then return tostring(result) end
    return nil
end

function serialize(inputCas, outputStream, parameters)
    local operation = parameter(parameters, "operation") or "pipeline"
    if operation ~= "pipeline" and operation ~= "extract" and operation ~= "mux" then
        error("Unsupported duui-video-anon operation")
    end

    local videos = JCasUtil:select(inputCas, Video):iterator()
    if not videos:hasNext() then error("No Video in source view") end
    local video = videos:next()
    if videos:hasNext() then error("Expected exactly one Video per CAS") end

    local audio = nil
    if operation == "mux" then
        local audioView = parameter(parameters, "audio_view") or "anonymized_audio"
        local ok, value = pcall(function()
            return inputCas:getView(audioView):getSofaDataString()
        end)
        if ok then audio = value end
        -- Some deployed speaker images use this fixed view instead of the
        -- DUUI target view used by the current repository source.
        if audio == nil and audioView == "anonymized_audio" then
            ok, value = pcall(function()
                return inputCas:getView("opf_anonymized_audio"):getSofaDataString()
            end)
            if ok then audio = value end
        end
        if audio == nil then error("No anonymized audio in speaker output view") end
    end

    outputStream:write(json.encode({
        operation = operation,
        video = {
            src = video:getSrc(),
            length = video:getLength(),
            fps = video:getFps(),
            mimetype = video:getMimetype(),
            begin = video:getBegin(),
            ["end"] = video:getEnd()
        },
        audio = audio,
        options = {
            anon_type = parameter(parameters, "anon_type"),
            redact_type = parameter(parameters, "redact_type"),
            language = parameter(parameters, "language"),
            hf_token = parameter(parameters, "hf_token")
        }
    }))
end

function deserialize(inputCas, inputStream)
    local body = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local result = json.decode(body)
    if result["operation"] == "extract" then
        inputCas:setSofaDataString(result["audio"] or "", "audio/wav")
    elseif result["operation"] == "mux" or result["operation"] == "pipeline" then
        local data = result["video"]
        if data == nil then error("Video response is missing") end
        local video = luajava.newInstance("org.texttechnologylab.annotation.type.Video", inputCas)
        video:setSrc(data["src"])
        video:setLength(data["length"])
        video:setFps(data["fps"])
        video:setMimetype("video/mp4")
        video:setBegin(data["begin"])
        video:setEnd(data["end"])
        video:addToIndexes()
    else
        error("Unexpected duui-video-anon response")
    end
end

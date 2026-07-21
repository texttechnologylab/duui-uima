-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

local function param_get(params, key)
    if params == nil then return nil end
    local v = params[key]
    if v ~= nil then return tostring(v) end
    local ok, r = pcall(function() return params:get(key) end)
    if ok and r ~= nil then return tostring(r) end
    return nil
end

local OPTION_KEYS = {"language"}

local function get_or_create_view(inputCas, name)
    local ok, view = pcall(function() return inputCas:getView(name) end)
    if ok and view ~= nil then
        return view
    end

    ok, view = pcall(function() return inputCas:createView(name) end)
    if ok and view ~= nil then
        return view
    end

    error("Unable to get or create CAS view: " .. name)
end

local function copy_options(params)
    local options = {}
    for _, key in ipairs(OPTION_KEYS) do
        local value = param_get(params, key)
        if value ~= nil then
            options[key] = value
        end
    end
    return options
end

-- Serialize the CAS into a JSON request sent to the Python service.
function serialize(inputCas, outputStream, params)
    local audio_b64 = inputCas:getSofaDataString()
    if audio_b64 == nil then audio_b64 = "" end

    local options = copy_options(params)
    local language = param_get(params, "language")
    if language == nil then
        language = inputCas:getDocumentLanguage()
        if language == nil or language == "x-unspecified" then
            language = ""
        end
    end
    if language ~= "" and language ~= nil then
        options["language"] = language
    end

    outputStream:write(json.encode({
        audio = audio_b64,
        options = options,
    }))
end

-- Deserialize the JSON response from the Python service back into the CAS.
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance(
        "java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    -- DUUI resolves .withTargetView(...) before deserialization, so inputCas
    -- is already the transcript view here.
    if results["original_text"] ~= nil then
        inputCas:setSofaDataString(results["original_text"], "text/plain")
    end

    -- Store the anonymized audio in a stable, reusable output view.
    if results["anonymized_audio"] ~= nil then
        local view = get_or_create_view(inputCas, "opf_anonymized_audio")
        view:setSofaDataString(results["anonymized_audio"], "audio/wav")
    end
end

-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    local audioBase64 = inputCas:getSofaDataString() --inputCas:getView(audioView):getSofaDataString()

    local model = params["model"]
    local batch_size = params["batch_size"]
    local allow_download = params["allow_download"]

    local language = params["language"]
    if language == nil then
        language = inputCas:getDocumentLanguage()

        if language == nil or language == "x-unspecified" then
            language = ""
        end
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        audio = audioBase64,
        language = language,
        model = model,
        batch_size = batch_size,
        allow_download = allow_download
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

    -- Add tokens to jcas
    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["audio_token"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local entireText = ""
        for i, sent in ipairs(results["audio_token"]) do
            if entireText == "" then
                entireText = entireText .. sent["text"]
            else
                entireText = entireText .. " " .. sent["text"]
            end

            local audioToken = luajava.newInstance("org.texttechnologylab.annotation.type.AudioToken", inputCas)
            audioToken:setBegin(sent["begin"])
            audioToken:setEnd(sent["end"])
            audioToken:setTimeStart(sent["timeStart"])
            audioToken:setTimeEnd(sent["timeEnd"])
            audioToken:setValue(sent["text"])
            audioToken:addToIndexes()

            local meta = results["meta"]
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(audioToken)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end

        inputCas:setSofaDataString(entireText, "text/plain")
    end

    if results["language"] ~= nil then
        -- set document language, especially needed as the result is most probably stored in a new, empty view
        if inputCas:getDocumentLanguage() == nil or inputCas:getDocumentLanguage() == "x-unspecified" then
            inputCas:setDocumentLanguage(results["language"])
        end
    end
end

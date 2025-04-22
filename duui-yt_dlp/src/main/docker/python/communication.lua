-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    -- Get output view
    local link = inputCas:getSofaDataString()

    -- Toggle YouTube transcription
    local withTranscription = params["withTranscription"]
    local cookies = params["cookies"]

        if cookies == nil then
        cookies = ""
    end

    if withTranscription == nil then
        withTranscription = false
    elseif withTranscription == "true" then
        withTranscription = true
    else
        withTranscription = false
    end

    local language = inputCas:getDocumentLanguage()

    if language == "x-unspecified" then
        language =  ""
    end

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        link = link,
        with_youtube_transcription = withTranscription,
        transcription_language = language,
        with_cookies = cookies
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
    inputCas:setSofaDataString(results["encoded_video"], results["mimetype_video"]);

    -- AudioToken generation
    if results["youtube_audio_token"] ~= nil and #results["youtube_audio_token"] > 0 then

        local transcriptionCas = inputCas:createView("yt_transcription")
        transcriptionCas:setDocumentText(results["youtube_audio_token"][1]["text"])
        for i, sent in ipairs(results["youtube_audio_token"]) do

            local audioToken = luajava.newInstance("org.texttechnologylab.annotation.type.AudioToken", transcriptionCas)
            audioToken:setBegin(sent["begin"])
            audioToken:setEnd(sent["end"])
            audioToken:setTimeStart(sent["timeStart"])
            audioToken:setTimeEnd(sent["timeEnd"])
            audioToken:setValue(sent["text"])
            audioToken:addToIndexes()
        end
    end
end

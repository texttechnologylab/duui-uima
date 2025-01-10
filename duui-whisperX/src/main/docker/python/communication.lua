-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    local audioBase64 = inputCas:getSofaDataString() --inputCas:getView(audioView):getSofaDataString()

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        audio = audioBase64,
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
    if results["audio_token"] ~= nil then

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

        end

        inputCas:setDocumentText(entireText)
    end
end

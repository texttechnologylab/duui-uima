-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    print("Start serialize")
    local videoBase64 = inputCas:getSofaDataString() --inputCas:getView(audioView):getSofaDataString()
    print ("Video ", videoBase64)
    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        video = videoBase64,
        language = language
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    --print("deserialize")
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

    --print("results", results)
    -- Add tokens to jcas
    if results["audio"] ~= nil then

        inputCas:setSofaDataString(results["audio"], "audio/mp3")
    end
end
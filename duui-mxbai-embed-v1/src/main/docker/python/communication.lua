-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
FloatArray = luajava.bindClass("org.apache.uima.jcas.cas.FloatArray")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)

    local text = inputCas:getSofaDataString()
    local chunkSize = params["chunkSize"] or 900

    -- ollamaConfig arrives as a raw string param; decode it into a table so it
    -- is serialised as a JSON object rather than a JSON string.
    local ollamaConfig = nil
    if params["ollamaConfig"] ~= nil then
        local raw = params["ollamaConfig"]
        -- Try direct decode first, then wrapped in braces (param may omit them)
        local ok, decoded = pcall(json.decode, raw)
        if not ok then
            ok, decoded = pcall(json.decode, "{" .. raw .. "}")
        end
        ollamaConfig = ok and decoded or nil
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        text = text,
        ollamaConfig = ollamaConfig,
        chunkSize = chunkSize
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
    if results["embeddings"] ~= nil then

        for i, sent in ipairs(results["embeddings"]) do

            local embedding = luajava.newInstance("org.texttechnologylab.uima.type.Embedding", inputCas)
            embedding:setBegin(sent["begin"])
            embedding:setEnd(sent["end"])

            local floatArray = FloatArray:create(inputCas, sent["embeddings"])

            embedding:setEmbedding(floatArray)
            embedding:addToIndexes()
        end
    end

end

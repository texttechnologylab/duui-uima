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

    local selection_type = params["selection"] ~= nil and params["selection"]
        or "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
    local clazz = Class:forName(selection_type);
    local selectionSet = JCasUtil:select(inputCas, clazz):iterator()
    local selection_array = {}

    while selectionSet:hasNext() do
        local s = selectionSet:next()

        local tSelection = {
            sText = s:getCoveredText(),
            iBegin = s:getBegin(),
            iEnd = s:getEnd()
        }
        table.insert(selection_array, tSelection)
    end

    local text = inputCas:getSofaDataString()
    local chunkSize = params["chunkSize"] or 900
    local apiUrl = params["apiUrl"] or ""
    local model = params["model"] or ""
    local apiKey = params["apiKey"] or ""


    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        apiUrl = apiUrl,
        selection = selection_array,
        text = text,
        model = model,
        apiKey = apiKey,
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

        local metaData = luajava.newInstance("org.texttechnologylab.annotation.MetaData", inputCas)
        metaData:setSource(results["source"])
        metaData:addToIndexes()

        for i, sent in ipairs(results["embeddings"]) do

            local embedding = luajava.newInstance("org.texttechnologylab.uima.type.Embedding", inputCas)
            embedding:setBegin(sent["begin"])
            embedding:setEnd(sent["end"])

            local floatArray = FloatArray:create(inputCas, sent["embeddings"])

            embedding:setEmbedding(floatArray)
            embedding:setModelReference(metaData)
            embedding:addToIndexes()
        end
    end

end

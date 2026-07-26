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
    -- Get data from CAS

    -- local clazz = Class:forName("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence");

    local clazz = Class:forName(params["type"]);

    local sentences = {}

    local sentences_it = JCasUtil:select(inputCas, clazz):iterator()
    while sentences_it:hasNext() do
        local sentence = sentences_it:next()
        local s = {
            text = sentence:getCoveredText(),
            begin = sentence:getBegin(),
            ['end'] = sentence:getEnd()
        }
        sentences[#sentences+1] = s
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        texts = sentences
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

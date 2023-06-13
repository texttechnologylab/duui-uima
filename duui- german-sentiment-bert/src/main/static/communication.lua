-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
utils = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local doc_text = inputCas:getDocumentText()
    local sentences = utils:select(inputCas, sentence):iterator()

    local sentences_array = {}

    while sentences:hasNext() do
        local s = sentences:next()
        local tSentence = {
            text = s:getCoveredText(),
            iBegin = s:getBegin(),
            iEnd = s:getEnd()
            }
        print(tSentence)
        table.insert(sentences_array, tSentence)
    end

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        doc_text = doc_text,
        doc_length = string.len(doc_text),
        sentences = sentences_array
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
    for i, pSentiment in ipairs(results["sentiments"]) do

        local sentiment = luajava.newInstance("org.texttechnologylab.annotation.SentimentBert", inputCas)

        sentiment:setBegin(pSentiment["iBegin"])
        sentiment:setEnd(pSentiment["iEnd"])
        sentiment:setProbabilityPositive(pSentiment["probabilityPositive"])
        sentiment:setProbabilityNeutral(pSentiment["probabilityNeutral"])
        sentiment:setProbabilityNegative(pSentiment["probabilityNegative"])
        sentiment:setSentiment(pSentiment["sentiment"])
        sentiment:addToIndexes()

     end

end

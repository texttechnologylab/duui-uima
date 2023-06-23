-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Paraphrase = luajava.bindClass("org.texttechnologylab.annotation.Paraphrase")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local doc_lang = inputCas:getDocumentLanguage()
    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    local cas_sentences = {}

    local sent_counter = 1
    local sents = util:select(inputCas, Sentence):iterator()
    while sents:hasNext() do
        local sent = sents:next()
        cas_sentences[sent_counter] = {}
        cas_sentences[sent_counter]["begin"] = sent:getBegin()
        cas_sentences[sent_counter]["end"] = sent:getEnd()
        cas_sentences[sent_counter]["coveredText"] = sent:getCoveredText()
        sent_counter = sent_counter + 1
    end

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        sentences = cas_sentences
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
    for i, sent_para in ipairs(results["paraphrases"]) do
        for j, para in ipairs(sent_para) do
            local paraphrase = luajava.newInstance("org.texttechnologylab.annotation.Paraphrase", inputCas)
            paraphrase:setBegin(para["begin"])
            paraphrase:setEnd(para["end"])
            paraphrase:setValue(para["paraphrased_text"])
            paraphrase:addToIndexes()
        end
    end

end
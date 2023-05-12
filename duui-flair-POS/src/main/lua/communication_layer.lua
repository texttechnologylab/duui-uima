-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
POS = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: A map of optional parameters
function serialize(inputCas, outputStream, parameters)
    -- Get data from CAS
    local document_text = inputCas:getDocumentText();

    local language = nil
    if parameters then
        language = parameters["language"]
    else
        print("No parameters were given, inferring language from CAS")
    end
    if language == nil then
        language = inputCas:getDocumentLanguage()
    end
    if language == "x-unspecified" or language == nil then
        error("Document language was not given and could not be inferred", 2)
    end

    local sentences = {}
    local sent_counter = 1
    local sents = JCasUtil:select(inputCas, Sentence):iterator()
    while sents:hasNext() do
        local sent = sents:next()
        local sent_begin = sent:getBegin()
        local sent_text = sent:getCoveredText()
        sentences[sent_counter] = {}
        sentences[sent_counter]["offset"] = sent_begin
        sentences[sent_counter]["coveredText"] = sent_text
        sent_counter = sent_counter + 1
    end
    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        text = document_text,
        language = language,
        sentences = sentences
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

    -- Add Taxa
    for i, tag in ipairs(results["tags"]) do
        local pos = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", inputCas)
        pos:setBegin(tag["begin"])
        pos:setEnd(tag["end"])
        pos:setPosValue(tag["pos_value"])
        pos:addToIndexes()
    end

end

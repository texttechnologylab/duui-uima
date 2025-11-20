-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
ConditionSentence = luajava.bindClass("org.texttechnologylab.annotation.neglab.ConditionSentence")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    local selection_array = {}

    local model = params["model"]

    local selectionSet = util:select(inputCas, ConditionSentence):iterator()

    while selectionSet:hasNext() do
        local s = selectionSet:next()

        local tSelection = {
            sText = s:getCoveredText(),
            iBegin = s:getBegin(),
            iEnd = s:getEnd(),
            sCondition = s:getCondition(),
            sTarget = s:getTarget()
        }
        table.insert(selection_array, tSelection)
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        selection = selection_array,
        model_name = model
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
    local model_name = results["model_name"]
    print(model_name)
    for i, suprise in ipairs(results["sentences"]) do

        local sentence = util:selectSingleAt(inputCas, ConditionSentence, suprise["iBegin"], suprise["iEnd"])

        if sentence ~= nil then
            sentence:setValue(suprise["sSuprise"])
            sentence:addToIndexes()

            local comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            comment:setReference(sentence)
            comment:setKey("model")
            comment:setValue(model_name)
            comment:addToIndexes()

        end
    end

end

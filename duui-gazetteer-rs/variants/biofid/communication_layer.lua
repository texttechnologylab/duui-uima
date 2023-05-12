-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: A map of optional parameters
function serialize(inputCas, outputStream, parameters)
    -- Get data from CAS
    local doc_text = inputCas:getDocumentText();
    if parameters then
        local max_len = parameters["max_len"]
        local result_selection = parameters["result_selection"]
        -- Encode data as JSON object and write to stream
        if max_len and result_selection then
            outputStream:write(json.encode({
                text = doc_text,
                max_len = max_len,
                result_selection = result_selection,
            }))
            return
        elseif max_len then
            outputStream:write(json.encode({
                text = doc_text,
                max_len = max_len,
            }))
            return
        elseif result_selection then
            outputStream:write(json.encode({
                text = doc_text,
                result_selection = result_selection,
            }))
            return
        end
    end
    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        text = doc_text
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
    for i, match in ipairs(results) do
        local taxon = luajava.newInstance("org.texttechnologylab.annotation.type.Taxon", inputCas)
        taxon:setValue(match["match_strings"])
        taxon:setIdentifier(match["match_labels"])
        taxon:setBegin(match["begin"])
        taxon:setEnd(match["end"])
        taxon:addToIndexes()
    end

end

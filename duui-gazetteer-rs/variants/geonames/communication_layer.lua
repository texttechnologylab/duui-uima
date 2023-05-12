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

    -- Add GeoNamesEntities
    for _, match in ipairs(results) do
        local matches = {}
        for str in string.gmatch(match["match_labels"], "([^|]+)") do
            table.insert(matches, str)
        end

        for _, _match in ipairs(matches) do
            local parts = {}
            for str in string.gmatch(_match:gsub("%s+", ""), "([^~]+)") do
                table.insert(parts, str)
            end

            local taxon = luajava.newInstance("org.texttechnologylab.annotation.GeoNamesEntity", inputCas)
            taxon:setBegin(match["begin"])
            taxon:setEnd(match["end"])

            taxon:setId(tonumber(parts[1]))
            taxon:setMainclass(parts[2])
            taxon:setSubclass(parts[3])
            taxon:addToIndexes()

            break -- FIXME: Currently only the first match is considered
        end
    end
end

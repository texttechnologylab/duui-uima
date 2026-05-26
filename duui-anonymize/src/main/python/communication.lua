-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

local function copy_options(params)
    local options = {}

    for key, value in pairs(params or {}) do
        if key ~= "selection" and key ~= "selection_begin" and key ~= "selection_end" and key ~= "selection_start" and key ~= "selection_stop" then
            options[key] = value
        end
    end

    return options
end

local function resolve_selection(params)
    if params == nil then
        return nil
    end

    local selection = params["selection"]
    if type(selection) == "table" then
        local begin = selection["begin"] or selection["start"]
        local ending = selection["end"] or selection["stop"]
        if begin ~= nil and ending ~= nil then
            return {
                begin = begin,
                ["end"] = ending,
            }
        end
    end

    local begin = params["selection_begin"] or params["selection_start"]
    local ending = params["selection_end"] or params["selection_stop"]
    if begin ~= nil and ending ~= nil then
        return {
            begin = begin,
            ["end"] = ending,
        }
    end

    return nil
end

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    local text = inputCas:getSofaDataString()
    if text == nil then
        text = ""
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        text = text,
        options = copy_options(params),
        selection = resolve_selection(params)
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

    local targetCas = inputCas
    if inputCas.createView ~= nil then
        local ok, view = pcall(function()
            return inputCas:createView("opf_redacted")
        end)
        if ok and view ~= nil then
            targetCas = view
        end
    end

    if results["redacted_text"] ~= nil then
        targetCas:setSofaDataString(results["redacted_text"], "text/plain")
    elseif results["text"] ~= nil then
        targetCas:setSofaDataString(results["text"], "text/plain")
    end

    if results["detected_spans"] ~= nil then
        for i, sent in ipairs(results["detected_spans"]) do
            local anomaly = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly", targetCas)
            anomaly:setBegin(sent["start"])
            anomaly:setEnd(sent["end"])
            anomaly:setCategory(sent["label"])
            anomaly:setDescription(sent["placeholder"] or sent["text"] or sent["label"])
            anomaly:addToIndexes()
        end
    end

    if results["selection"] ~= nil then
        -- Selection metadata is available in the JSON response for downstream consumers.
    end
end

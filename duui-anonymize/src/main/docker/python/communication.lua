-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- Known option keys forwarded to the Python service.
-- pairs() does not iterate Java map objects in LuaJ, so we read each key explicitly.
local OPTION_KEYS = {
    "mode", "model", "device",
    "context_window_length", "trim_whitespace",
    "output_mode", "discard_overlapping_predicted_spans",
}

local function copy_options(params)
    if params == nil then return {} end
    local options = {}
    for _, key in ipairs(OPTION_KEYS) do
        local value = params[key]
        if value ~= nil then
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
        local begin  = selection["begin"] or selection["start"]
        local ending = selection["end"]   or selection["stop"]
        if begin ~= nil and ending ~= nil then
            return { begin = begin, ["end"] = ending }
        end
    end

    local begin  = params["selection_begin"] or params["selection_start"]
    local ending = params["selection_end"]   or params["selection_stop"]
    if begin ~= nil and ending ~= nil then
        return { begin = begin, ["end"] = ending }
    end

    return nil
end

-- Serialize the CAS into a JSON request sent to the Python service.
function serialize(inputCas, outputStream, params)
    local text = inputCas:getSofaDataString()
    if text == nil then text = "" end

    outputStream:write(json.encode({
        text      = text,
        options   = copy_options(params),
        selection = resolve_selection(params),
    }))
end

-- Deserialize the JSON response from the Python service back into the CAS.
--
-- Anomaly annotations are added to the *original* CAS view so their
-- character offsets remain valid against the original document text.
-- The redacted text is stored as the sofa of a separate "opf_redacted" view.
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    -- Store redacted text in its own view (offsets here belong to redacted text,
    -- so we do NOT add Anomaly annotations to this view).
    if results["redacted_text"] ~= nil then
        local ok, view = pcall(function() return inputCas:createView("opf_redacted") end)
        if ok and view ~= nil then
            view:setSofaDataString(results["redacted_text"], "text/plain")
        end
    end

    -- Add Anomaly annotations to the original view; offsets reference original text.
    if results["detected_spans"] ~= nil then
        for i, span in ipairs(results["detected_spans"]) do
            local anomaly = luajava.newInstance(
                "de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly", inputCas)
            anomaly:setBegin(span["start"])
            anomaly:setEnd(span["end"])
            anomaly:setCategory(span["label"])
            -- description = replacement used (e.g. "[private_person]") or original word
            anomaly:setDescription(span["placeholder"] ~= "" and span["placeholder"]
                                   or span["text"] or span["label"])
            anomaly:addToIndexes()
        end
    end
end

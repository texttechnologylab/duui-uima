-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- Read a parameter from params regardless of whether it is a Lua table or a
-- LuaJ-wrapped Java Map.  Direct table indexing works for Lua tables; Java
-- Map objects (HashMap, etc.) require params:get(key) instead.
local function param_get(params, key)
    if params == nil then return nil end
    local v = params[key]
    if v ~= nil then return tostring(v) end
    local ok, r = pcall(function() return params:get(key) end)
    if ok and r ~= nil then return tostring(r) end
    return nil
end

-- Known option keys forwarded to the Python service.
local OPTION_KEYS = {
    "mode", "model", "device",
    "context_window_length", "trim_whitespace",
    "output_mode", "discard_overlapping_predicted_spans",
}

local function copy_options(params)
    local options = {}
    print("Copying options:")
    for _, key in ipairs(OPTION_KEYS) do
        local value = param_get(params, key)
        if value ~= nil then
            print("  ", key, "=", value)
            options[key] = value
        end
    end
    return options
end

local function resolve_selection(params)
    if params == nil then return nil end

    -- selection passed as a nested table
    local selection = params["selection"]
    if selection == nil then
        local ok, r = pcall(function() return params:get("selection") end)
        if ok then selection = r end
    end
    if type(selection) == "table" then
        local b = selection["begin"] or selection["start"]
        local e = selection["end"]   or selection["stop"]
        if b ~= nil and e ~= nil then
            return { begin = b, ["end"] = e }
        end
    end

    -- selection passed as flat begin/end keys
    local b = param_get(params, "selection_begin") or param_get(params, "selection_start")
    local e = param_get(params, "selection_end")   or param_get(params, "selection_stop")
    if b ~= nil and e ~= nil then
        return { begin = tonumber(b), ["end"] = tonumber(e) }
    end

    return nil
end

-- Serialize the CAS into a JSON request sent to the Python service.
function serialize(inputCas, outputStream, params)
    local text = inputCas:getSofaDataString()
    if text == nil then text = "" end

    local options = copy_options(params)

    outputStream:write(json.encode({
        text      = text,
        options   = options,
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

    -- Store redacted text in its own view.
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
            anomaly:setDescription(
                (span["placeholder"] ~= nil and span["placeholder"] ~= "") and span["placeholder"]
                or span["text"] or span["label"])
            anomaly:addToIndexes()
        end
    end
end

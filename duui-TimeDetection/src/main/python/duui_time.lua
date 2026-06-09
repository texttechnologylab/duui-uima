StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

DEFAULT_SELECTION = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

-- Runtime defaults. These mirror the Python/ENV defaults and can be overwritten
-- with DUUI .withParameter(...).
DEFAULT_THRESHOLD = 0.0
DEFAULT_BATCH_SIZE = 8

DEFAULT_TIMEX_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.TimeX3"
DATE_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.time.Date"
TIME_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.time.Time"
DURATION_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.time.Duration"
SET_TYPE = "org.texttechnologylab.annotation.semaf.isotimeml.time.Set"

function get_parameter(parameters, key, default_value)
    if parameters ~= nil and parameters[key] ~= nil then
        return parameters[key]
    end
    return default_value
end

function safe_string(value)
    if value == nil then
        return ""
    end
    return tostring(value)
end

function parse_optional_number(value, default_value, parameter_name, min_value, max_value)
    if value == nil then
        return default_value
    end

    local parsed = tonumber(value)
    if parsed == nil then
        error("Parameter '" .. parameter_name .. "' must be a number", 2)
    end

    if min_value ~= nil and parsed < min_value then
        error("Parameter '" .. parameter_name .. "' must be >= " .. tostring(min_value), 2)
    end

    if max_value ~= nil and parsed > max_value then
        error("Parameter '" .. parameter_name .. "' must be <= " .. tostring(max_value), 2)
    end

    return parsed
end

function trim(value)
    return string.gsub(safe_string(value), "^%s*(.-)%s*$", "%1")
end

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)

    local selection_types = get_parameter(parameters, "selection", DEFAULT_SELECTION)

    local threshold = parse_optional_number(
        get_parameter(parameters, "threshold", DEFAULT_THRESHOLD),
        DEFAULT_THRESHOLD,
        "threshold",
        0.0,
        1.0
    )

    local batch_size = parse_optional_number(
        get_parameter(parameters, "batch_size", DEFAULT_BATCH_SIZE),
        DEFAULT_BATCH_SIZE,
        "batch_size",
        1,
        nil
    )
    batch_size = math.floor(batch_size)

    local document_creation_time = trim(get_parameter(parameters, "document_creation_time", ""))
    if document_creation_time == "" then
        document_creation_time = trim(get_parameter(parameters, "reference_time", ""))
    end

    local duckling_url = trim(get_parameter(parameters, "duckling_url", ""))
    local corenlp_url = trim(get_parameter(parameters, "corenlp_url", ""))
    local duckling_timezone = trim(get_parameter(parameters, "duckling_timezone", "Europe/Berlin"))

    local selections = {}
    local selections_count = 1

    for selection_type in string.gmatch(selection_types, "([^,]+)") do
        selection_type = trim(selection_type)

        local sentences = {}
        local sentences_count = 1

        if selection_type == "text" then
            sentences[1] = {
                text = doc_text,
                begin = 0,
                ['end'] = doc_len
            }
        else
            local clazz = Class:forName(selection_type)
            local sentences_it = JCasUtil:select(inputCas, clazz):iterator()

            while sentences_it:hasNext() do
                local sentence = sentences_it:next()
                sentences[sentences_count] = {
                    text = sentence:getCoveredText(),
                    begin = sentence:getBegin(),
                    ['end'] = sentence:getEnd()
                }
                sentences_count = sentences_count + 1
            end
        end

        selections[selections_count] = {
            sentences = sentences,
            selection = selection_type
        }
        selections_count = selections_count + 1
    end

    outputStream:write(json.encode({
        selections = selections,
        lang = doc_lang,
        doc_len = doc_len,
        threshold = threshold,
        batch_size = batch_size,
        document_creation_time = document_creation_time,
        duckling_url = duckling_url,
        corenlp_url = corenlp_url,
        duckling_timezone = duckling_timezone
    }))
end

function add_document_modification(inputCas, results)
    if results["modification_meta"] == nil then
        return
    end

    pcall(function()
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(safe_string(modification_meta["user"]))
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(safe_string(modification_meta["comment"]))
        modification_anno:addToIndexes()
    end)
end

function add_model_metadata(inputCas, results)
    local model_meta = nil

    pcall(function()
        model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(safe_string(results["model_version"]))
        model_meta:setModelName(safe_string(results["model_name"]))
        model_meta:setSource(safe_string(results["model_source"]))
        model_meta:setLang(safe_string(results["model_lang"]))
        model_meta:addToIndexes()
    end)

    return model_meta
end

function add_annotation_comment(inputCas, reference, key, value)
    if value == nil then
        return
    end

    pcall(function()
        local comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
        comment:setReference(reference)
        comment:setKey(safe_string(key))
        comment:setValue(safe_string(value))
        comment:addToIndexes()
    end)
end

function get_time_type_from_tag(tag)
    local tag_type = tag["time_type"]
    if tag_type ~= nil and tag_type ~= "" then
        return tag_type
    end

    local timex_type = string.upper(safe_string(tag["timex_type"]))
    if timex_type == "DATE" then
        return DATE_TYPE
    elseif timex_type == "TIME" then
        return TIME_TYPE
    elseif timex_type == "DURATION" then
        return DURATION_TYPE
    elseif timex_type == "SET" then
        return SET_TYPE
    end

    return DEFAULT_TIMEX_TYPE
end

function create_time_annotation(inputCas, tag)
    local tag_type = get_time_type_from_tag(tag)
    local annotation = nil

    local ok = pcall(function()
        annotation = luajava.newInstance(tag_type, inputCas)
    end)

    if not ok or annotation == nil then
        annotation = luajava.newInstance(DEFAULT_TIMEX_TYPE, inputCas)
    end

    annotation:setBegin(tag["begin"])
    annotation:setEnd(tag["end"])

    local value = tag["value"]
    if value == nil then
        value = tag["timex_value"]
    end

    pcall(function()
        annotation:setValue(safe_string(value))
    end)

    if tag["function_in_document"] ~= nil then
        pcall(function()
            annotation:setFunctionInDocument(safe_string(tag["function_in_document"]))
        end)
    end

    if tag["temporal_function"] ~= nil then
        pcall(function()
            annotation:setTemporalFunction(tag["temporal_function"])
        end)
    end

    if tag["quant"] ~= nil then
        pcall(function()
            annotation:setQuant(safe_string(tag["quant"]))
        end)
    end

    if tag["freq"] ~= nil then
        pcall(function()
            annotation:setFreq(safe_string(tag["freq"]))
        end)
    end

    annotation:addToIndexes()
    return annotation
end

function get_tags(results)
    if results["tags"] ~= nil then
        return results["tags"]
    end

    -- Fallback for flattened response fields.
    local tags = {}
    local begins = results["begin"] or {}
    local ends = results["end"] or {}
    local timex_types = results["results"] or {}
    local timex_values = results["timex_value"] or {}
    local time_types = results["time_type"] or {}
    local covered_texts = results["covered_text"] or {}
    local factors = results["factors"] or {}
    local models = results["model"] or {}

    for i, timex_type in ipairs(timex_types) do
        tags[i] = {
            begin = begins[i],
            ['end'] = ends[i],
            value = timex_values[i],
            timex_type = timex_type,
            time_type = time_types[i],
            covered_text = covered_texts[i],
            score = factors[i],
            model_name = models[i]
        }
    end

    return tags
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results == nil then
        return
    end

    add_document_modification(inputCas, results)
    add_model_metadata(inputCas, results)

    local tags = get_tags(results)

    for i, tag in ipairs(tags) do
        if tag["begin"] ~= nil and tag["end"] ~= nil then
            local annotation = create_time_annotation(inputCas, tag)

            -- Optional comments for traceability.
            add_annotation_comment(inputCas, annotation, "score", tag["score"])
            add_annotation_comment(inputCas, annotation, "covered_text", tag["covered_text"])
            add_annotation_comment(inputCas, annotation, "model_name", tag["model_name"])
            add_annotation_comment(inputCas, annotation, "timex_type", tag["timex_type"])
            add_annotation_comment(inputCas, annotation, "time_type", tag["time_type"])
        end
    end
end

StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

DEFAULT_SELECTION = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

-- Runtime defaults. These mirror the Python/ENV defaults and can be overwritten
-- with DUUI .withParameter(...).
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_LABELS = "person,organization,location,date,event,product,taxon,other"

DEFAULT_NER_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
TAXON_TYPE = "org.texttechnologylab.annotation.type.Taxon"

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

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)

    local selection_types = get_parameter(parameters, "selection", DEFAULT_SELECTION)

    -- Runtime parameters. Defaults are declared here and mirrored in Python.
    -- DUUI parameters override these defaults.
    local threshold_parameter = get_parameter(parameters, "threshold", DEFAULT_THRESHOLD)
    local threshold = DEFAULT_THRESHOLD
    if threshold_parameter ~= nil then
        threshold = tonumber(threshold_parameter)
        if threshold == nil then
            error("Parameter 'threshold' must be a number between 0.0 and 1.0", 2)
        end
        if threshold < 0.0 or threshold > 1.0 then
            error("Parameter 'threshold' must be between 0.0 and 1.0", 2)
        end
    end

    local batch_size_parameter = get_parameter(parameters, "batch_size", DEFAULT_BATCH_SIZE)
    local batch_size = DEFAULT_BATCH_SIZE
    if batch_size_parameter ~= nil then
        batch_size = tonumber(batch_size_parameter)
        if batch_size == nil then
            error("Parameter 'batch_size' must be a positive integer", 2)
        end
        batch_size = math.floor(batch_size)
        if batch_size < 1 then
            error("Parameter 'batch_size' must be greater than or equal to 1", 2)
        end
    end

    local labels = get_parameter(parameters, "labels", DEFAULT_LABELS)
    labels = safe_string(labels)
    labels = string.gsub(labels, "^%s*(.-)%s*$", "%1")
    if labels == "" then
        error("Parameter 'labels' must contain at least one label", 2)
    end

    local selections = {}
    local selections_count = 1

    for selection_type in string.gmatch(selection_types, "([^,]+)") do
        selection_type = string.gsub(selection_type, "^%s*(.-)%s*$", "%1")

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
        labels = labels
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

function create_ner_annotation(inputCas, tag)
    local tag_type = tag["ner_type"]
    local value = string.lower(safe_string(tag["value"]))

    -- If Python returns a taxon label, prefer the TTLab Taxon type.
    -- This also covers older responses where ner_type was missing or still NamedEntity.
    if value == "taxon" or value == "taxa" then
        tag_type = TAXON_TYPE
    elseif tag_type == nil or tag_type == "" then
        tag_type = DEFAULT_NER_TYPE
    end

    local annotation = nil

    -- Prefer the concrete NER subtype returned by Python, e.g. Person, Location, Organization, Taxon.
    local ok = pcall(function()
        annotation = luajava.newInstance(tag_type, inputCas)
    end)

    -- Fallback to generic NamedEntity if the subtype is not available in the active type system.
    if not ok or annotation == nil then
        annotation = luajava.newInstance(DEFAULT_NER_TYPE, inputCas)
    end

    annotation:setBegin(tag["begin"])
    annotation:setEnd(tag["end"])

    -- DKPro NamedEntity and TTLab NamedEntity-like types usually provide setValue,
    -- but keep this safe for custom types without this feature.
    pcall(function()
        annotation:setValue(safe_string(tag["value"]))
    end)

    if tag["identifier"] ~= nil then
        pcall(function()
            annotation:setIdentifier(safe_string(tag["identifier"]))
        end)
    end

    annotation:addToIndexes()
    return annotation
end

function get_tags(results)
    if results["tags"] ~= nil then
        return results["tags"]
    end

    -- Fallback for the flattened response fields of duui_ner_single_model.py.
    local tags = {}
    local begins = results["begin"] or {}
    local ends = results["end"] or {}
    local labels = results["results"] or {}
    local ner_types = results["ner_type"] or {}
    local covered_texts = results["covered_text"] or {}
    local factors = results["factors"] or {}
    local models = results["model"] or {}

    for i, label in ipairs(labels) do
        tags[i] = {
            begin = begins[i],
            ['end'] = ends[i],
            value = label,
            ner_type = ner_types[i],
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
    local model_meta = add_model_metadata(inputCas, results)

    local tags = get_tags(results)

    for i, tag in ipairs(tags) do
        if tag["begin"] ~= nil and tag["end"] ~= nil then
            local annotation = create_ner_annotation(inputCas, tag)

            -- These comments are optional. They are only written if the TTLab AnnotationComment
            -- type exists in the active type system.
            add_annotation_comment(inputCas, annotation, "score", tag["score"])
            add_annotation_comment(inputCas, annotation, "covered_text", tag["covered_text"])
            add_annotation_comment(inputCas, annotation, "model_name", tag["model_name"])
            add_annotation_comment(inputCas, annotation, "ner_type", tag["ner_type"])
        end
    end
end
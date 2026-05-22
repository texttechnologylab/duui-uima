-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
Float = luajava.bindClass("java.lang.Float")
NeerMatchPrediction = luajava.bindClass("org.texttechnologylab.annotation.NeerMatchPrediction")

function extractNeProperties(inputCas, entity)
    local properties = {}
    local identifier = entity:getIdentifier()
    if identifier then
        properties.identifier = identifier
    end
    local value = entity:getValue()
    if value then
        properties.value = value
    end
    return properties
end

function extractTokenProperties(inputCas, token)
    local properties = {}
    local pos = token:getPos()
    if pos then
        local posValue = pos:getPosValue()
        if posValue then
            properties.pos = posValue
        end
        local coarseValue = pos:getCoarseValue()
        if coarseValue then
            properties.coarse = coarseValue
        end
    end
    local stem = token:getStemValue()
    if stem then
        properties.stem = stem
    end
    local lemma = token:getLemmaValue()
    if lemma then
        properties.lemma = lemma
    end
    local form = token:getFormValue()
    if form then
        properties.form = form
    end
    return properties
end

function serialize(inputCas, outputStream, parameters)
    local pipeline_id = parameters["pipeline_id"]
    local entities = {}
    local selection_name = parameters["selection"]
    local selection_class = Class:forName(selection_name)
    local selection_iterator = JCasUtil:select(inputCas, selection_class):iterator()
    local is_named_entity = selection_name == "de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity"
    local is_token = selection_name == "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
    while selection_iterator:hasNext() do
        local entity = selection_iterator:next()
        local entity_text = entity:getCoveredText()
        local entity_id = entity:getAddress()
        local result_entity = {
            entity_id = entity_id,
            text = entity_text
        }
        if is_named_entity then
            local properties = extractNeProperties(inputCas, entity)
            result_entity.properties = properties
        elseif is_token then
            local properties = extractTokenProperties(inputCas, entity)
            result_entity.properties = properties
        end
        table.insert(entities, result_entity)
    end

    outputStream:write(json.encode({
        pipeline_id = pipeline_id,
        entities = entities,
        selection = selection_name
    }))
end

function deserialize(inputCas, inputStream)
    local inputBytes = inputStream:readAllBytes()
    local inputString = luajava.newInstance("java.lang.String", inputBytes, StandardCharsets.UTF_8)
    local data = json.decode(inputString)
    local pipeline_id = data.pipeline_id
    local stored_index = data.stored_index
    local stored_count = data.stored_count
    print("Stored " .. stored_count .. " entities for pipeline " .. pipeline_id .. " with document index " .. stored_index)
end

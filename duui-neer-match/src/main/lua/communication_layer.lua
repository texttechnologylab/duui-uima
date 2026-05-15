-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
Float = luajava.bindClass("java.lang.Float")
NeerMatchPrediction = luajava.bindClass("org.texttechnologylab.annotation.NeerMatchPrediction")

function serialize(inputCas, outputStream, parameters)
    local pipeline_id = parameters["pipeline_id"]
    local entities = {}
    local selection_class = Class:forName(parameters["selection"])
    local selection_iterator = JCasUtil:select(inputCas, selection_class):iterator()
    local entity_count = 0
    while selection_iterator:hasNext() do
        local entity = selection_iterator:next()
        local entity_text = entity:getCoveredText()
        local entity_id = entity_count
        entity_count = entity_count + 1
        table.insert(entities, {
            text = entity_text,
            entity_id = entity_id
        })
    end

    outputStream:write(json.encode({
        pipeline_id = pipeline_id,
        entities = entities
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

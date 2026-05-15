-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
Float = luajava.bindClass("java.lang.Float")

function serialize(inputCas, outputStream, parameters)
    local model_name = parameters["model"]
    local entities = {}
    local selection_class = Class:forName(parameters["selection"])
    local selection_iterator = JCasUtil:select(inputCas, selection_class):iterator()
    while selection_iterator:hasNext() do
        local entity = selection_iterator:next()
        local entity_text = entity:getCoveredText()
        local entity_begin = entity:getBegin()
        local entity_end = entity:getEnd()
        local entity_id = entity:getId()
        local user_data = {
            begin = entity_begin,
            endd = entity_end,
            id = entity_id,
            type = parameters["selection"]
        }
        table.insert(entities, {
            text = entity_text,
            user_data = user_data
        })
    end
    local query = parameters["query"]
    local batch_size = 32
    local limit = 5
    local query_entities = {}
    table.insert(query_entities, {
        text = query
    })
    local properties = {
        limit = limit,
        batch_size = batch_size,
        model = model_name
    }
    if parameters["threshold"] then
        properties.threshold = Float:valueOf(parameters["threshold"])
    end

    outputStream:write(json.encode({
        entities = query_entities,
        targets = entities,
        properties = properties
    }))
end

function deserialize(inputCas, inputStream)
    local inputBytes = inputStream:readAllBytes()
    local inputString = luajava.newInstance("java.lang.String", inputBytes, StandardCharsets.UTF_8)
    local data = json.decode(inputString)
    local results = data.results
    if #results ~= 1 then
        print("Expected exactly one result, got " .. #results)
    end
    local result = results[1]
    local query = result.entity
    local suggestions = result.suggestions
    for _, suggestion in ipairs(suggestions) do
        local entityText = suggestion.target
        local entityIndex = suggestion.target_index
        local score = suggestion.score
        local userData = suggestion.target_user_data
        local begin = userData.begin
        local endd = userData.endd
        local entityId = userData.id
        local entityType = userData.type
        local entity = JCasUtil:selectByIndex(inputCas, Class:forName(entityType), entityIndex)
        if entity == nil then
            print("Could not find entity for index " .. entityIndex)
        else
            -- Here you can add the logic to handle the matched entity, e.g., add annotations or update the CAS as needed
            print("Matched entity: " .. entityText .. " with score: " .. score)
        end
    end
end

-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
Float = luajava.bindClass("java.lang.Float")

function serialize(inputCas, outputStream, parameters)
    local pipeline_id = parameters["pipeline_id"]
    local do_store = inputCas:getDocumentText() ~= nil and #inputCas:getDocumentText() > 0
    if do_store then
        local entities = {}
        local selection_class = Class:forName(parameters["selection"])
        local selection_iterator = JCasUtil:select(inputCas, selection_class):iterator()
        while selection_iterator:hasNext() do
            local entity = selection_iterator:next()
            local entity_text = entity:getCoveredText()
            local entity_id = entity:getId()
            table.insert(entities, {
                text = entity_text,
                entity_id = entity_id
            })
        end

        outputStream:write(json.encode({
            action = "store",
            pipeline_id = pipeline_id,
            entities = query_entities
        }))
    else
        local model_name = parameters["model"]
        local batch_size = 32
        local properties = {
            batch_size = batch_size,
            model = model_name
        }
        if parameters["threshold"] then
            properties.threshold = Float:valueOf(parameters["threshold"])
        end

        outputStream:write(json.encode({
            action = "process",
            pipeline_id = pipeline_id,
            properties = properties,
            clear_storage = true
        }))
    end
end

function deserialize(inputCas, inputStream)
    local inputBytes = inputStream:readAllBytes()
    local inputString = luajava.newInstance("java.lang.String", inputBytes, StandardCharsets.UTF_8)
    local data = json.decode(inputString)
    if data.action == "store" then
        local pipeline_id = data.pipeline_id
        local stored_index = data.stored_index
        local stored_count = data.stored_count
        print("Stored " .. stored_count .. " entities for pipeline " .. pipeline_id .. " with document index " .. stored_index)
    else
        local pipeline_id = data.pipeline_id
        local results = data.results
        for _, comparison_result in ipairs(results) do
            local document_1_index = comparison_result.document_1_index
            local document_2_index = comparison_result.document_2_index
            local predictions = comparison_result.predictions
            print("Received " .. #predictions .. " predictions for pipeline " .. pipeline_id .. " comparing document " .. document_1_index .. " and document " .. document_2_index)

            local result_view = inputCas:createView("Result_" .. document_1_index .. "_" .. document_2_index)
            for _, prediction in ipairs(predictions) do
                local document_1_entity = prediction.document_1_entity
                local document_2_entity = prediction.document_2_entity
                local score = prediction.score
                local prediction_anno = luajava.newInstance("org.texttechnologylab.annotation.NeerMatchPrediction", result_view)
                prediction_anno:setDocument1EntityText(document_1_entity.text)
                prediction_anno:setDocument2EntityText(document_2_entity.text)
                prediction_anno:setDocument1EntityId(document_1_entity.entity_id)
                prediction_anno:setDocument2EntityId(document_2_entity.entity_id)
                prediction_anno:setScore(score)
                prediction_anno:addToIndexes()
            end
        end
    end
    --[[ local results = data.results
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
    end ]]
end

StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, parameters)
    outputStream:write(json.encode({
        types = parameters["types"],
        exact = parameters["exact"],
    }))
end

-- Removes all annotations of the requested types from the indexes. Note: with a DUUI segmentation strategy the
-- removal is not merged back into the document (only new annotations are), so this component must run without one.
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    local cas = inputCas:getCas()
    local type_system = cas:getTypeSystem()
    local exact = results["exact"]

    local removed = {}
    for i, type_name in ipairs(results["types"]) do
        local uima_type = type_system:getType(type_name)
        if uima_type == nil then
            error("Type to remove is not in the type system: " .. type_name)
        end

        -- collect first, removing while iterating would invalidate the iterator
        local to_remove = luajava.newInstance("java.util.ArrayList")
        local fs_it = cas:getIndexRepository():getAllIndexedFS(uima_type)
        while fs_it:hasNext() do
            local fs = fs_it:next()
            if not exact or fs:getType():getName() == type_name then
                to_remove:add(fs)
            end
        end

        for j = 0, to_remove:size() - 1 do
            cas:removeFsFromIndexes(to_remove:get(j))
        end
        removed[#removed + 1] = type_name .. ": " .. to_remove:size()
    end

    local meta = results["meta"]
    local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
    modification_anno:setUser(meta["name"])
    modification_anno:setTimestamp(results["timestamp"])
    modification_anno:setComment(meta["name"] .. " (" .. meta["version"] .. "), removed"
            .. (exact and " (exact type)" or " (with subtypes)") .. ": " .. table.concat(removed, ", "))
    modification_anno:addToIndexes()
end

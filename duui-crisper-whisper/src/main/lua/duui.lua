StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, params)
    local audioBase64 = inputCas:getSofaDataString()

    outputStream:write(json.encode({
        audio = audioBase64,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["audio_token"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local entireText = ""
        for i, sent in ipairs(results["audio_token"]) do
            if entireText == "" then
                entireText = entireText .. sent["text"]
            else
                entireText = entireText .. " " .. sent["text"]
            end

            local audioToken = luajava.newInstance("org.texttechnologylab.annotation.type.AudioToken", inputCas)
            audioToken:setBegin(sent["begin"])
            audioToken:setEnd(sent["end"])
            audioToken:setTimeStart(sent["timeStart"])
            audioToken:setTimeEnd(sent["timeEnd"])
            audioToken:setValue(sent["text"])
            audioToken:addToIndexes()

            local meta = results["meta"]
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(audioToken)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end

        inputCas:setSofaDataString(entireText, "text/plain")
    end
end

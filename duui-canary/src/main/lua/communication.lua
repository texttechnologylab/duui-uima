StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, params)
    local audio = inputCas:getSofaDataString()

    local language = params["language"] or inputCas:getDocumentLanguage()
    local model = params["model"]

    outputStream:write(json.encode({
        audio = audio,
        language = language,
        model = model,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["audio_tokens"] ~= nil and results["audio_segments"] ~= nil and results["full_text"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        inputCas:setDocumentText(results["full_text"])

        for i, sent in ipairs(results["audio_tokens"]) do
            local audio_token = luajava.newInstance("org.texttechnologylab.annotation.type.AudioToken", inputCas)
            audio_token:setBegin(sent["begin"])
            audio_token:setEnd(sent["end"])
            audio_token:setTimeStart(sent["timeStart"])
            audio_token:setTimeEnd(sent["timeEnd"])
            audio_token:setValue(sent["text"])
            audio_token:addToIndexes()

            local meta = results["meta"]
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(audio_token)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end

        for i, sent in ipairs(results["audio_segments"]) do
            local audio_segment = luajava.newInstance("org.texttechnologylab.annotation.type.AudioSentence", inputCas)
            audio_segment:setBegin(sent["begin"])
            audio_segment:setEnd(sent["end"])
            audio_segment:setTimeStart(sent["timeStart"])
            audio_segment:setTimeEnd(sent["timeEnd"])
            audio_segment:addToIndexes()

            local meta = results["meta"]
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(audio_segment)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

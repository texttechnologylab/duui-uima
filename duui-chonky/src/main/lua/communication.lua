StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, parameters)
    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        keep_line_breaks = parameters["keep_line_breaks"],
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    local modification_meta = results["modification_meta"]
    local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
    modification_anno:setUser(modification_meta["user"])
    modification_anno:setTimestamp(modification_meta["timestamp"])
    modification_anno:setComment(modification_meta["comment"])
    modification_anno:addToIndexes()

    local meta = results["meta"]
    for i, paragraph in ipairs(results["paragraphs"]) do
        local paragraph_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph", inputCas)
        paragraph_anno:setBegin(paragraph["begin"])
        paragraph_anno:setEnd(paragraph["end"])
        paragraph_anno:addToIndexes()

        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
        meta_anno:setReference(paragraph_anno)
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:addToIndexes()
    end
end

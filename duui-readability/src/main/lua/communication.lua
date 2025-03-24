StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    outputStream:write(json.encode({
        lang = doc_lang,
        text = doc_text,
        begin = 0,
        ["end"] = doc_len,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["results"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]
        for i, result in ipairs(results["results"]) do
            local category_anno = luajava.newInstance("org.hucompute.textimager.uima.type.category.CategoryCoveredTagged", inputCas)
            category_anno:setValue(result["value"])
            category_anno:setScore(result["score"])
            category_anno:setTags(result["tags"])
            category_anno:setBegin(result["begin"])
            category_anno:setEnd(result["end"])
            category_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(category_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

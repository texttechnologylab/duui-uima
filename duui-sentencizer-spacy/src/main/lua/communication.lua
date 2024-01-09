StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    outputStream:write(json.encode({
        text = doc_text,
        len = doc_len,
        lang = doc_lang
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["sentences"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]
        for j, sentence in ipairs(results["sentences"]) do
            local sent_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", inputCas)
            sent_anno:setBegin(sentence["begin"])
            sent_anno:setEnd(sentence["end"])
            sent_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(sent_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end
    end
end

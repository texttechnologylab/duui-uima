StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")

function serialize(inputCas, outputStream, parameters)
    -- every sentence is tokenized separately, so no token crosses a sentence boundary
    local sentences = {}
    local sentence_list = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Sentence))
    for i = 0, sentence_list:size() - 1 do
        local sentence = sentence_list:get(i)
        sentences[#sentences + 1] = {
            begin = sentence:getBegin(),
            ['end'] = sentence:getEnd(),
        }
    end

    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        len = DUUIUtils:getDocumentTextLength(inputCas),
        lang = inputCas:getDocumentLanguage(),
        sentences = sentences,
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
    for i, token in ipairs(results["tokens"]) do
        local token_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", inputCas)
        token_anno:setBegin(token["begin"])
        token_anno:setEnd(token["end"])
        token_anno:addToIndexes()

        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
        meta_anno:setReference(token_anno)
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:addToIndexes()
    end
end

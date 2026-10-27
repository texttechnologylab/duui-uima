StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Paragraph = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph")

function serialize(inputCas, outputStream, parameters)
    -- every paragraph is split separately, so no sentence crosses a paragraph boundary
    local paragraphs = {}
    local paragraph_list = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Paragraph))
    for i = 0, paragraph_list:size() - 1 do
        local paragraph = paragraph_list:get(i)
        paragraphs[#paragraphs + 1] = {
            begin = paragraph:getBegin(),
            ['end'] = paragraph:getEnd(),
        }
    end

    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        len = DUUIUtils:getDocumentTextLength(inputCas),
        paragraphs = paragraphs,
        split_on_input_newlines = parameters["split_on_input_newlines"],
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
    for i, sentence in ipairs(results["sentences"]) do
        local sentence_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", inputCas)
        sentence_anno:setBegin(sentence["begin"])
        sentence_anno:setEnd(sentence["end"])
        sentence_anno:addToIndexes()

        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
        meta_anno:setReference(sentence_anno)
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:addToIndexes()
    end
end

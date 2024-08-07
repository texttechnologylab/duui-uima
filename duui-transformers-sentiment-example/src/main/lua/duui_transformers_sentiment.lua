StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
SentimentUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    -- Get document text, language and size from CAS
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = SentimentUtils:getDocumentTextLength(inputCas)

    -- Encode as JSON and write to the output stream, this is then sent to the tool
    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        doc_len = doc_len,
    }))
end

function deserialize(inputCas, inputStream)
    -- Read the JSON from the input stream and decode it
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    -- Check if the JSON contains the expected keys...
    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["sentiment_label"] ~= nil and results["sentiment_score"] ~= nil then
        -- Create the modification meta annotation
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        -- Create the sentiment metadata
        local sentiment_anno = luajava.newInstance("org.hucompute.textimager.uima.type.Sentiment", inputCas)
        sentiment_anno:setBegin(0)
        sentiment_anno:setEnd(SentimentUtils:getDocumentTextLength(inputCas))
        sentiment_anno:setSentiment(sentiment_label)
        sentiment_anno:addToIndexes()

        -- Create the annotation metadata, note that it references the sentiment annotation
        local meta = results["meta"]
        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
        meta_anno:setReference(sentiment_anno)
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:addToIndexes()
    end
end

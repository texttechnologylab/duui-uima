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
    if results["sentiment_label"] ~= nil and results["sentiment_score"] ~= nil then
        -- Create the sentiment metadata
        local sentiment_anno = luajava.newInstance("org.hucompute.textimager.uima.type.Sentiment", inputCas)
        sentiment_anno:setBegin(0)
        sentiment_anno:setEnd(SentimentUtils:getDocumentTextLength(inputCas))
        sentiment_anno:setSentiment(sentiment_label)
        sentiment_anno:addToIndexes()
    end
end

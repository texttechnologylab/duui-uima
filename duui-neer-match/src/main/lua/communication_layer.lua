-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

function serialize(inputCas, outputStream, parameters)
    -- for now just pass the document text to the output stream
    local doc_text = inputCas:getDocumentText()
    local doc_lang = inputCas:getDocumentLanguage()

    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang
    }))
end

function deserialize(inputCas, inputStream)
    local inputBytes = inputStream:readAllBytes()
    local inputString = luajava.newInstance("java.lang.String", inputBytes, StandardCharsets.UTF_8)
    local data = json.decode(inputString)
    inputCas:setDocumentText(data.text)
end

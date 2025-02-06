StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Base64 = luajava.bindClass("java.util.Base64")

PDF_MIME_TYPE = "application/pdf"

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()

    local min_chars = parameters["min_chars"]
    local ocr_dpi = parameters["ocr_dpi"]
    local ocr_preprocess = parameters["ocr_preprocess"]

    -- check mimetype but do not fail automatically
    local mime_type = inputCas:getSofaMimeType()
    if mime_type ~= PDF_MIME_TYPE then
        print("Warning: SofA mimetype is not \"" .. PDF_MIME_TYPE .. "\"")
    end

    -- send PDF bytes base64 encoded
    local data = Base64:getEncoder():encodeToString(inputCas:getSofaDataStream():readAllBytes());

    outputStream:write(json.encode({
        lang = doc_lang,
        data = data,
        min_chars = min_chars,
        ocr_dpi = ocr_dpi,
        ocr_preprocess = ocr_preprocess
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["text"] ~= nil then
        -- Set extracted text as SofA string
        -- Note: we expect this view to be empty, else this will fail
        local text = results["text"]
        inputCas:setDocumentText(text)

        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]
        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
        meta_anno:setReference(inputCas:getSofa())
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:addToIndexes()
    end
end

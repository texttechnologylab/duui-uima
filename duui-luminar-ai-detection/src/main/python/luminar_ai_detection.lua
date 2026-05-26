StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
DUUILuaUtils     = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

-- UIMA type to instantiate
local AIDetectionType = "org.texttechnologylab.annotation.luminar.AIDetection"

function serialize(inputCas, outputStream, parameters)
    -- Get document text, language and size from CAS
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText() or ""

    local doc_len = DUUILuaUtils:getDocumentTextLength(inputCas)

    -- Encode as JSON and write to the output stream (this goes to the tool)
    outputStream:write(json.encode({
        text    = doc_text,
        lang    = doc_lang,
        doc_len = doc_len,
    }))
end

function deserialize(inputCas, inputStream)
    -- Read DUUI response JSON
    local inputString = luajava.newInstance(
        "java.lang.String",
        inputStream:readAllBytes(),
        StandardCharsets.UTF_8
    )
    local results = json.decode(inputString)
    if results == nil or results["detections"] == nil then
        return
    end

    local doc_text = inputCas:getDocumentText() or ""
    local doc_len  = #doc_text

    -- Iterate detections and write annotations
    for i, det in ipairs(results["detections"]) do
        local b = det["begin"] or 0
        local e = det["end"]   or 0

        -- Clamp to document length
        if b < 0 then b = 0 end
        if e < 0 then e = 0 end
        if b > doc_len then b = doc_len end
        if e > doc_len then e = doc_len end
        if e < b then e = b end

        -- Create annotation instance
        local anno = luajava.newInstance(AIDetectionType, inputCas)
        anno:setBegin(b)
        anno:setEnd(e)
        anno:setDetectionScore(det["detectionScore"]) -- 0..1
        anno:setLevel(det["level"])                   -- "SEQUENCE" | "DOCUMENT"
        anno:setModel(det["model"])                   -- model HF id

        anno:addToIndexes()
    end
end

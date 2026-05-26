StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUILuaUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    --     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUILuaUtils:getDocumentTextLength(inputCas)

    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        doc_len = doc_len,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    for i, speaker in ipairs(results["speakers"]) do

        local _speaker = luajava.newInstance("org.texttechnologylab.annotation.parliament.Speaker", inputCas)
        _speaker:setBegin(speaker["begin"])
        _speaker:setEnd(speaker["end"])
        _speaker:setLabel(speaker["label"]);
        _speaker:setFirstname(speaker["firstname"])
        _speaker:setName(speaker["name"])
        _speaker:setName(speaker["nobility"])
        _speaker:setName(speaker["title"])
        _speaker:setName(speaker["role"])
        _speaker:setName(speaker["party"])
        _speaker:setName(speaker["party_deducted"])
        _speaker:setName(speaker["electoral_county"])
        _speaker:setName(speaker["electoral_county_deducted"])
        _speaker:addToIndexes()

    end

    for i, speech in ipairs(results["speeches"]) do

        local _speech = luajava.newInstance("org.texttechnologylab.annotation.parliament.Speech", inputCas)
        _speech:setBegin(speech["begin"])
        _speech:setEnd(speech["end"])
        _speech:addToIndexes()

    end

end

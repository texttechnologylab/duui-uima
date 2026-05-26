StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
--     print(model_name)
--     print(select)

    outputStream:write(json.encode({
        lang = doc_lang,
        text = doc_text,
        begin = 0,
        ["end"] = doc_len,
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
     local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
     local results = json.decode(inputString)
     if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["values"] ~= nil then
 --         print("GetInfo")
         local source = results["model_source"]
         local model_version = results["model_version"]
         local model_name = results["model_name"]
         local model_lang = results["model_lang"]
 --         print("meta")
         local modification_meta = results["modification_meta"]
         local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
         modification_anno:setUser(modification_meta["user"])
         modification_anno:setTimestamp(modification_meta["timestamp"])
         modification_anno:setComment(modification_meta["comment"])
         modification_anno:addToIndexes()

 --         print("setMetaData")
         local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
         model_meta:setModelVersion(model_version)
 --         print(model_version)
         model_meta:setModelName(model_name)
 --         print(model_name)
         model_meta:setSource(source)
 --         print(source)
         model_meta:setLang(model_lang)
 --         print(model_lang)
         model_meta:addToIndexes()

         local meta = results["meta"]
         local begin_read = results["begin"]
         local end_read = results["end"]
         local values = results["values"]
         local codes = results["codes"]
         local definitions = results["definitions"]
         local measures = results["measures"]
         local typeName = results["typeName"]
         local typeNumber = results["typeNumber"]

         for i, value in ipairs(values) do
             local l2sca_anno = luajava.newInstance("org.texttechnologylab.annotation.L2SCA", inputCas)
             l2sca_anno:setBegin(begin_read)
             l2sca_anno:setEnd(end_read)
             l2sca_anno:setValue(value)
             l2sca_anno:setCode(codes[i])
             l2sca_anno:setDefinition(definitions[i])
             l2sca_anno:setMeasure(measures[i])
             l2sca_anno:setTypeName(typeName[i])
             l2sca_anno:setTypeNumber(typeNumber[i])
             l2sca_anno:setModel(model_meta)
             l2sca_anno:addToIndexes()
         end
     end
 end

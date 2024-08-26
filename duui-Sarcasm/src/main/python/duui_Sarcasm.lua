StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Sarcasm = luajava.bindClass("org.texttechnologylab.annotation.Sarcasm")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
--     print(model_name)
    local selection_types = parameters["selection"]
--     print(select)

    local selections = {}
    local selections_count = 1
    for selection_type in string.gmatch(selection_types, "([^,]+)") do
       local sentences = {}
       if selection_type == "text" then
           local s = {
               text = doc_text,
               begin = 0,
               ['end'] = doc_len
           }
           sentences[1] = s
       else
--            print("start")
           local sentences_count = 1
           local clazz = Class:forName(selection_type);
           local sentences_it = JCasUtil:select(inputCas, clazz):iterator()
           while sentences_it:hasNext() do
               local sentence = sentences_it:next()
               local s = {
                   text = sentence:getCoveredText(),
                   begin = sentence:getBegin(),
                   ['end'] = sentence:getEnd()
               }
               print(sentence:getCoveredText())
               sentences[sentences_count] = s
               sentences_count = sentences_count + 1
           end
       end

       local selection = {
           sentences = sentences,
           selection = selection_type
       }
       selections[selections_count] = selection
       selections_count = selections_count + 1
    end
--     print("end")

    outputStream:write(json.encode({
        selections = selections,
        lang = doc_lang,
        doc_len = doc_len
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
--     print("start")
    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["begins"] ~= nil then
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
--         print("meta")
        local begins = results["begins"]
--         print("begins")
        local ends = results["ends"]
--         print("ends")
        local sarcasms = results["sarcasm"]
--         print("sarcasm")
        local non_sarcasms = results["non_sarcasm"]
--         print("non_sarcasm")
        for index_i, res in ipairs(sarcasms) do
            local sarcasm = luajava.newInstance("org.texttechnologylab.annotation.Sarcasm", inputCas)
            sarcasm:setBegin(begins[index_i])
--             print(begins[index_i])
            sarcasm:setEnd(ends[index_i])
--             print(ends[index_i])
            sarcasm:setSarcasm(sarcasms[index_i])
--             print(sarcasms[index_i])
            sarcasm:setNonSarcasm(non_sarcasms[index_i])
--             print(non_sarcasms[index_i])
            sarcasm:setModel(model_meta)
            sarcasm:addToIndexes()
        end
    end
end

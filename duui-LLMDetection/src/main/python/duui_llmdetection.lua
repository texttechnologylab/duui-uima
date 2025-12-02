StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)


    local selection_types = parameters["selection"]

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

    outputStream:write(json.encode({
        selections = selections,
        lang = doc_lang,
        doc_len = doc_len,
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
--          print(model_version)
         model_meta:setModelName(model_name)
--          print(model_name)
         model_meta:setSource(source)
--          print(source)
         model_meta:setLang(model_lang)
--          print(model_lang)
         model_meta:addToIndexes()

         local meta = results["meta"]
         local begin_read = results["begin"]
         local end_read = results["end"]
         local values = results["values"]
         local keys = results["keys"]
         local definitions = results["definitions"]
         local len_results = results["len_results"]


         for i, value in ipairs(values) do
             local begin_i = begin_read[i]
--              print(begin_i)
             local end_i = end_read[i]
             local len_i = len_results[i]

             local value_i = values[i]
             local key_i = keys[i]
--              print(end_i)
             local def_i = definitions[i]
--              print(def_i)

             for j, key_j in ipairs(key_i) do
--                  print(j)
                 local llm_detect = luajava.newInstance("org.texttechnologylab.annotation.LLMMetric", inputCas)
                 value_j = value_i[j]
                 key_j = key_i[j]
--                  print(key_j)
                 def_j = def_i[j]
                 llm_detect:setBegin(begin_i)
                 llm_detect:setEnd(end_i)
                 llm_detect:setValue(value_j)
                 llm_detect:setKeyName(key_j)
                 llm_detect:setDefinition(def_j)
                 llm_detect:setModel(model_meta)
                 llm_detect:addToIndexes()
             end
         end
     end
 end

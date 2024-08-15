StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)

    local model_name = parameters["model_name"]
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
        model_name = model_name
    }))
end

function deserialize(inputCas, inputStream)
     local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
     local results = json.decode(inputString)
     if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["results"] ~= nil then
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
         local begin_topic = results["begin"]
 --         print("begin_emo")
         local end_topic = results["end"]
 --         print("end_emo")
         local res_out = results["results"]
 --         print("results")
         local res_len = results["len_results"]
 --         print("Len_results")
         local factors = results["factors"]
 --         print(factors)
         for index_i, res in ipairs(res_out) do
 --             print(res)
             local begin_topic_i = begin_topic[index_i]
 --             print(begin_topic_i)
             local end_topic_i = end_topic[index_i]
 --             print(end_topic_i)
             local len_i = res_len[index_i]
 --             print(len_i)
 --             print(type(len_i))
             local topic_i = luajava.newInstance("org.texttechnologylab.annotation.Topic", inputCas, begin_topic_i, end_topic_i)
 --             print(topic_i)
             local fsarray = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, len_i)
 --             print(fsarray)
             topic_i:setTopics(fsarray)
             local counter = 0
             local factor_i = factors[index_i]
 --             print(factor_i)
             for index_j, topic_j in ipairs(res) do
 --                 print(topic_j)
                 local factor_j = factor_i[index_j]
 --                 print(factor_j)
                 topic_in_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                 topic_in_i:setReference(topic_i)
                 topic_in_i:setKey(topic_j)
                 topic_in_i:setValue(factor_j)
                 topic_in_i:addToIndexes()
                 topic_i:setTopics(counter, topic_in_i)
                 counter = counter + 1
             end
             topic_i:setModel(model_meta)
             topic_i:addToIndexes()
 --             print("add")
         end
     end
 --     print("end")
 end

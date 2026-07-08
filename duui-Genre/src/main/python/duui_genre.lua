StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)

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
        doc_len = doc_len
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
         local begin_genre = results["begin"]
 --         print("begin_emo")
         local end_genre = results["end"]
 --         print("end_emo")
         local res_out = results["results"]
--          print("results")
         local res_len = results["len_results"]
 --         print("Len_results")
         local factors = results["factors"]
--          print(factors)
         for index_i, res in ipairs(res_out) do
 --             print(res)
             local begin_genre_i = begin_genre[index_i]
 --             print(begin_genre_i)
             local end_genre_i = end_genre[index_i]
 --             print(end_genre_i)
             local len_i = res_len[index_i]
 --             print(len_i)
 --             print(type(len_i))
             local genre_i = luajava.newInstance("org.texttechnologylab.annotation.Genre", inputCas, begin_genre_i, end_genre_i)
 --             print(genre_i)
             local fsarray = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, len_i)
 --             print(fsarray)
             genre_i:setGenres(fsarray)
             local counter = 0
             local factor_i = factors[index_i]
 --             print(factor_i)
             for index_j, genre_j in ipairs(res) do
 --                 print(genre_j)
                 local factor_j = factor_i[index_j]
 --                 print(factor_j)
                 genre_in_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                 genre_in_i:setReference(genre_i)
                 genre_in_i:setKey(genre_j)
                 genre_in_i:setValue(factor_j)
                 genre_in_i:addToIndexes()
                 genre_i:setGenres(counter, genre_in_i)
                 counter = counter + 1
             end
             genre_i:setModel(model_meta)
             genre_i:addToIndexes()
 --             print("add")
         end
     end
 --     print("end")
 end

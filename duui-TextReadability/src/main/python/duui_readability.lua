StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, params)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
--     print(model_name)
--     print(select)

    local par = {
        homogenization = true,
        compression = true,
        ngram = 4
    }

    if params["diversity.homogenization"] ~= nil then
        par.homogenization = params["diversity.homogenization"]
    end

    if params["diversity.compression"] ~= nil then
        par.compression = params["diversity.compression"]
    end

    if params["diversity.ngram"] ~= nil then
        par.ngram = params["diversity.ngram"]
    end

    outputStream:write(json.encode({
        lang = doc_lang,
        text = doc_text,
        begin = 0,
        ["end"] = doc_len,
        params = par
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
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
         local begin_read = results["begin"]
         local end_read = results["end"]
         local res_out = results["results"]
         --         print("results")
         local res_len = results["len_results"]
         --         print("Len_results")
         local factors = results["factors"]
         local group_name = results["group_name"]
         local group_set_name = ""



         for index_i, group_i in ipairs(group_name) do
             local res_len_i = res_len[index_i]
             local factors_i = factors[index_i]
             local res_out_i = res_out[index_i]
             local read_i = luajava.newInstance("org.texttechnologylab.annotation.ReadabilityAdvance", inputCas)
             local fsarray = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, res_len_i)
             local counter = 0
             read_i:setTextReadabilities(fsarray)
             read_i:setGroupName(group_i)
             for index_j, res_j in ipairs(res_out_i) do
                 local factor_j = factors_i[index_j]
                 read_in_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                 read_in_i:setReference(read_i)
                 read_in_i:setKey(res_j)
                 read_in_i:setValue(factor_j)
                 read_in_i:addToIndexes()
                 read_i:setTextReadabilities(counter, read_in_i)
                 counter = counter + 1
             end
             read_i:setModel(model_meta)
             read_i:addToIndexes()
         end
     end
 --     print("end")
 end

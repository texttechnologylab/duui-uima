StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
SentimentUtitls = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = SentimentUtitls:getDocumentTextLength(inputCas)

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
        local begins = results["begin"]
--        print("begins")
        local ends = results["end"]
--        print("ends")
        local factors = results["factors"]
        local res_output = results["results"]
        local negative = results["negative"]
        local positive = results["positive"]
        local neutral = results["neutral"]
        for index_i, res in ipairs(res_output) do
            local sentiment_i = luajava.newInstance("org.texttechnologylab.annotation.SentimentModel", inputCas)
            sentiment_i:setBegin(begins[index_i])
            sentiment_i:setEnd(ends[index_i])
            local factor_i = factors[index_i]
            for index_j, sentiment_j in ipairs(res) do
--                 print(sentiment_j)
                local factor_j = factor_i[index_j]
                if sentiment_j == "negative" then
                    sentiment_i:setProbabilityNegative(factor_j)
                elseif sentiment_j == "positive" then
                    sentiment_i:setProbabilityPositive(factor_j)
                elseif sentiment_j == "neutral" then
                    sentiment_i:setProbabilityNeutral(factor_j)
                elseif sentiment_j == "sentiment" then
                    sentiment_i:setSentiment(factor_j)
                end
            end
            sentiment_i:setModel(model_meta)
            sentiment_i:addToIndexes()
        end
     end
 --     print("end")
 end

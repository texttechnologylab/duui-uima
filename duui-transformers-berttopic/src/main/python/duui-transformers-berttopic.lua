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
    if results["results"] ~= nil then
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

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
 --          print("Len_results")
         local factors = results["factors"]
 --         print(factors)
         local words = results["words"]
  --          print("Len_results")
         local words_prob = results["words_prob"]
         for index_i, res in ipairs(res_out) do
 --             print(res)
             local begin_topic_i = begin_topic[index_i]
 --             print(begin_topic_i)
             local end_topic_i = end_topic[index_i]
 --             print(end_topic_i)
             local len_i = res_len[index_i]
 --            print(len_i)
 --             print(type(len_i))
             local topic_i = luajava.newInstance("org.texttechnologylab.annotation.UnifiedTopic", inputCas, begin_topic_i, end_topic_i)
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
                 topic_in_i = luajava.newInstance("org.texttechnologylab.annotation.TopicValueBaseWithScore", inputCas)

                 topic_in_i:setScore(factor_j)

                 topic_in_i:setValue(topic_j)


                 words_for_topic_j = words[index_i][index_j]
                 num_words_in_topic_j = #words[index_i][index_j]
                 local counter_w = 0
                 local fsarray_word = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, num_words_in_topic_j)

                 topic_in_i:setWords(fsarray_word)
                    for index_k, word_k in ipairs(words_for_topic_j) do
                        word_in_i = luajava.newInstance("org.texttechnologylab.annotation.TopicWord", inputCas)
                        word_in_i:setWord(word_k)
                        word_in_i:setProbability(words_prob[index_i][index_j][index_k])
                        word_in_i:setTopic(topic_in_i)
                        word_in_i:addToIndexes()
                        topic_in_i:setWords(counter_w, word_in_i)
                        counter_w = counter_w + 1
                    end

                 topic_in_i:addToIndexes()
                 topic_i:setTopics(counter, topic_in_i)
                 counter = counter + 1
             end
             topic_i:setMetadata(model_meta)
             topic_i:addToIndexes()
 --             print("add")
         end
     end
 --     print("end")
 end
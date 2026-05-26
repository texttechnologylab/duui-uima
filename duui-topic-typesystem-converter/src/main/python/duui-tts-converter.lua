StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function get_topics(selection, selection_type)
    --print(selection)
    local topics = selection:getTopics()
    --print(topics)
    local topics_it = topics:iterator()
    local tops = {}
    local topics_count = 1
    while topics_it:hasNext() do
        local topic = topics_it:next()

        if selection_type == "org.texttechnologylab.annotation.BertTopic" then
            --print(selection_type)
            --print(topic:getValue())
            local s = {
                                value = topic:getValue(),
                                score = topic:getProbability(),
                            }
            tops[topics_count] = s
            topics_count = topics_count + 1
        else
            --print(selection_type)
            --print(topic:getValue())
            local s = {
                                value = topic:getKey(),
                                score = topic:getValue(),
                            }
            tops[topics_count] = s
            topics_count = topics_count + 1
        end
    end
    return tops
end

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)

    --local selection_type = parameters["selection"]
    local selection_types = parameters["selection"]
    local remove = parameters["remove_old"]

    --local sents = getSentences(inputCas)
    --local paras = getParagraphs(inputCas)


    selections = {}
    local selection_count = 1
    for selection_type in string.gmatch(selection_types, "([^;]+)") do
        print(selection_type)
        local anns ={}
        local ann_count = 1
        local clazz = Class:forName(selection_type);
        selection_it = JCasUtil:select(inputCas, clazz):iterator()
        while selection_it:hasNext() do
            local selection = selection_it:next()
            if selection_type == "org.hucompute.textimager.uima.type.category.CategoryCoveredTagged" then

                local s = {
                           value = selection:getValue(),
                           score = selection:getScore(),
                           tags = selection:getTags(),
                          begin = selection:getBegin(),
                          ['end'] = selection:getEnd()
                       }
               anns[ann_count] = s
               ann_count = ann_count + 1
            end
            if selection_type == "org.texttechnologylab.annotation.Topic" or selection_type == "org.texttechnologylab.annotation.BertTopic" then
                topics = get_topics(selection, selection_type)



                anns[ann_count] = {
                    begin= selection:getBegin(),
                    ['end'] = selection:getEnd(),
                    topics = topics,
                }
                ann_count = ann_count + 1
            end

        end

        selections[selection_count] = {
                type = selection_type,
                annotations = anns,
            }
        selection_count = selection_count + 1



        if remove == "true" then

            local old_annotations = JCasUtil:select(inputCas, Class:forName(selection_type))
            for i = 0, old_annotations:size() - 1 do
                old_annotations:get(i):removeFromIndexes()
            end
            if selection_type == "org.texttechnologylab.annotation.Topic" then
                local old_topics = JCasUtil:select(inputCas, Class:forName('org.texttechnologylab.annotation.AnnotationComment'))
                for i = 0, old_topics:size() - 1 do
                    old_topics:get(i):removeFromIndexes()
                end
            end

        end
    end



    outputStream:write(json.encode({
       --sentences = sents,
       --paragraphs = paras,
       doc_text = doc_text,
       selections = selections,
       doc_lang = doc_lang,
       doc_len = doc_len,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    if results["model_name"] ~=nil then
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
    end
    if results["results"] ~= nil then


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
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
-- switch to new utils, needs DUUI version >= 9d7a3e7c581ae7977763262397a4f533c6d7edec and rebuilt Dockers
SentimentUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
--SentimentUtils = luajava.bindClass("org.hucompute.textimager.uima.transformers.sentiment.Utils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = SentimentUtils:getDocumentTextLength(inputCas)

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

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["selections"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]

        for i, selection in ipairs(results["selections"]) do
            local selection_type = selection["selection"]
            for j, sentence in ipairs(selection["sentences"]) do
                -- simple sentiment
                local sentiment_anno = luajava.newInstance("org.hucompute.textimager.uima.type.Sentiment", inputCas)
                sentiment_anno:setBegin(sentence["sentence"]["begin"])
                sentiment_anno:setEnd(sentence["sentence"]["end"])
                sentiment_anno:setSentiment(sentence["sentiment"])
                sentiment_anno:addToIndexes()

                local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
                meta_anno:setReference(sentiment_anno)
                meta_anno:setName(meta["name"])
                meta_anno:setVersion(meta["version"])
                meta_anno:setModelName(meta["modelName"])
                meta_anno:setModelVersion(meta["modelVersion"])
                meta_anno:addToIndexes()

                local meta_selection = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                meta_selection:setReference(sentiment_anno)
                meta_selection:setKey("selection")
                meta_selection:setValue(selection_type)
                meta_selection:addToIndexes()

                local meta_score = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                meta_score:setReference(sentiment_anno)
                meta_score:setKey("score")
                meta_score:setValue(sentence["score"])
                meta_score:addToIndexes()

                -- raw label scores
                for label, score in pairs(sentence["details"]) do
                    local raw_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                    raw_anno:setReference(sentiment_anno)
                    raw_anno:setKey("raw_score")
                    raw_anno:setValue(label .. "=" .. score)
                    raw_anno:addToIndexes()
                end

                -- detailed sentiment
                local sentiment_anno2 = luajava.newInstance("org.hucompute.textimager.uima.type.CategorizedSentiment", inputCas)
                sentiment_anno2:setBegin(sentence["sentence"]["begin"])
                sentiment_anno2:setEnd(sentence["sentence"]["end"])
                sentiment_anno2:setSentiment(sentence["polarity"])
                sentiment_anno2:setPos(sentence["pos"])
                sentiment_anno2:setNeu(sentence["neu"])
                sentiment_anno2:setNeg(sentence["neg"])
                sentiment_anno2:addToIndexes()

                local meta_anno2 = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
                meta_anno2:setReference(sentiment_anno2)
                meta_anno2:setName(meta["name"])
                meta_anno2:setVersion(meta["version"])
                meta_anno2:setModelName(meta["modelName"])
                meta_anno2:setModelVersion(meta["modelVersion"])
                meta_anno2:addToIndexes()

                local meta_selection2 = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                meta_selection2:setReference(sentiment_anno2)
                meta_selection2:setKey("selection")
                meta_selection2:setValue(selection_type)
                meta_selection2:addToIndexes()

                -- add address of simple annotation to each detailed
                local meta_simple = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                meta_simple:setReference(sentiment_anno2)
                meta_simple:setKey("sentiment_ref")
                meta_simple:setValue(sentiment_anno:getAddress())
                meta_simple:addToIndexes()
            end
        end
    end
end

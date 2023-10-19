-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentences = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
msgpack = luajava.bindClass("org.msgpack.core.MessagePack")
comments = luajava.bindClass("org.texttechnologylab.annotation.AnnotationComment")
anomaly = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly")
suggest = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SuggestedAction")
anomalyspelling = luajava.bindClass("org.texttechnologylab.annotation.AnomlySpelling")
anomalyspellingMeta = luajava.bindClass("org.texttechnologylab.annotation.AnomalySpellingMeta")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local doc_text = inputCas:getDocumentText()
    local doc_lang = inputCas:getDocumentLanguage()
--     print(doc_text)
--     print(doc_lang)
    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    local sentences_cas = {}
    local token_cas = {}
    local sen_counter = 1
--     print("start")
    local sents = util:select(inputCas, Sentences):iterator()
    while sents:hasNext() do
        local sent = sents:next()
        local begin_sen = sent:getBegin()
        local end_sen = sent:getEnd()
        -- token to sentence
        local token_counter = 1
        local tokens = util:selectCovered(Token, sent):iterator()
        sentences_cas[sen_counter] = {}
        sentences_cas[sen_counter]["begin"] = begin_sen
        sentences_cas[sen_counter]["end"] = end_sen
        token_cas[sen_counter] = {}
        while tokens:hasNext() do
            local token = tokens:next()
            local token_begin = token:getBegin()
            local token_end = token:getEnd()
            token_cas[sen_counter][token_counter] = {}
            token_cas[sen_counter][token_counter]["begin"] = token_begin
            token_cas[sen_counter][token_counter]["end"] = token_end
            token_cas[sen_counter][token_counter]["text"] = token:getCoveredText()
            token_counter = token_counter + 1
        end
        sen_counter = sen_counter + 1
    end
--     print("sentences")
    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        sen = sentences_cas,
        tokens = token_cas
    }))
--     print("sendToPython")
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    -- Parse JSON data from string into object
    local results = json.decode(inputString)


     -- Add modification annotation
     local modification_meta = results["modification_meta"]
     local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
     modification_anno:setUser(modification_meta["user"])
     modification_anno:setTimestamp(modification_meta["timestamp"])
     modification_anno:setComment(modification_meta["comment"])
     modification_anno:addToIndexes()


    -- Get meta data, this is the same for every annotation
     local meta = results["meta"]
     local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
     meta_anno:setReference(modification_anno)
     meta_anno:setName(meta["name"])
     meta_anno:setVersion(meta["version"])
     meta_anno:setModelName(meta["modelName"])
     meta_anno:setModelVersion(meta["modelVersion"])
     meta_anno:addToIndexes()
--      print("Start")


--     Get Token results
    local sen_document = results["tokens"]
--
--
        for i, sent in ipairs(sen_document) do
            -- read Tokens
            for j, token in ipairs(sent) do
                -- Control every Spelling which can be correct with Symspell
--                 print(token["spellout"])
                if token["spellout"]=="wrong" or token["spellout"]=="unknown" or token["spellout"]=="skipped" or token["spellout"]=="right" then
                    -- counter for the suggestion Anomaly can save x SuggestedAction
--                     print(token["begin"])
--                     print(token["end"])
--                     print(token["suggestion"])
                    local counter_suggest = 0
                    local spellout_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SuggestedAction", inputCas)
                    spellout_anno:setBegin(token["begin"])
                    spellout_anno:setEnd(token["end"])
                    spellout_anno:setReplacement(token["suggestion"])
                    spellout_anno:setCertainty(1.0)
                    spellout_anno:addToIndexes()
-- --                     print("suggestion")
--
--                     local spellout_anomaly = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly", inputCas)
--                     spellout_anomaly:setBegin(token["begin"])
--                     spellout_anomaly:setEnd(token["end"])
--                     spellout_anomaly:setSuggestions(luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, 1))
--                     spellout_anomaly:setSuggestions(counter_suggest, spellout_anno)
--                     spellout_anomaly:setCategory("Symspell")
--                     spellout_anomaly:addToIndexes()
--                     print("Anomaly")

                    local anomly_spelling = luajava.newInstance("org.texttechnologylab.annotation.AnomlySpelling", inputCas)
                    anomly_spelling:setBegin(token["begin"])
                    anomly_spelling:setEnd(token["end"])
                    anomly_spelling:setSuggestions(luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, 1))
                    anomly_spelling:setSuggestions(counter_suggest, spellout_anno)
                    anomly_spelling:setSpellingType(token["spellout"])
                    anomly_spelling:setModelName("Symspell")
                    anomly_spelling:setCategory("Symspell")
                    anomly_spelling:addToIndexes()
--                     print("SpellingAnomly")

--                     local anno_comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
--                     anno_comment:setReference(spellout_anno)
--                     anno_comment:setKey("Spelling")
--                     anno_comment:setValue("Symspell")
--                     anno_comment:addToIndexes()
--
--                     local anno_comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
--                     anno_comment:setReference(spellout_anno)
--                     anno_comment:setKey("SpellingType")
--                     anno_comment:setValue(token["spellout"])
--                     anno_comment:addToIndexes()
--                     print("comment")
                end

                if token["spellout"]=="meta" then
                    local anomaly_spelling_meta = luajava.newInstance("org.texttechnologylab.annotation.AnomalySpellingMeta", inputCas)
                    anomaly_spelling_meta:setModelName("Symspell")
--                     print("Model")
                    anomaly_spelling_meta:setGoodQuality(token["goodQuality"])
                    anomaly_spelling_meta:setUnknownQuality(token["unknownQuality"])
                    anomaly_spelling_meta:setQuality(token["quality"])
--                     print("Quality")

                    anomaly_spelling_meta:setRightWords(token["right"])
                    anomaly_spelling_meta:setWrongWords(token["wrong"])
                    anomaly_spelling_meta:setSkippedWords(token["skipped"])
                    anomaly_spelling_meta:setUnknownWords(token["unknown"])
--                     print("Words")

                    anomaly_spelling_meta:setPercentRight(token["percentRight"])
                    anomaly_spelling_meta:setPercentWrong(token["percentWrong"])
                    anomaly_spelling_meta:setPercentUnknown(token["percentUnknown"])
--                     print("Percent")

                    anomaly_spelling_meta:setPercentRightWithoutSkipped(token["percentRightWithoutSkipped"])
                    anomaly_spelling_meta:setPercentWrongWithoutSkipped(token["percentWrongWithoutSkipped"])
                    anomaly_spelling_meta:setPercentUnknownWithoutSkipped(token["percentUnknownWithoutSkipped"])
--                     print("PercentWithoutSkipped")

                    anomaly_spelling_meta:addToIndexes()
                end
    --
    --             -- Control every Spelling which can not be correct with Symspell, that spellings will be corrected with BERT
--                 if token["spellout"]=="unknown" then
--                     -- Anomaly
--                     local spellout_anomaly = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly", inputCas)
--                     spellout_anomaly:setBegin(token["begin"])
--                     spellout_anomaly:setEnd(token["end"])
--                     spellout_anomaly:setSuggestions(luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, 3))
--                     spellout_anomaly:addToIndexes()
-- --                     print("Anomaly")
--                     local counter_suggest = 0
--                     for model_name, k in pairs(token["toolPred"]) do
--                         print(model_name)
--                         local certainty_i = token["toolPred"][model_name]["probability"]
--                         local suggestion_i = token["toolPred"][model_name]["word"]
--                         local spellout_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.SuggestedAction", inputCas)
--                         spellout_anno:setBegin(token["begin"])
--                         spellout_anno:setEnd(token["end"])
--                         spellout_anno:setReplacement(suggestion_i)
--                         spellout_anno:setCertainty(certainty_i)
--                         spellout_anno:addToIndexes()
-- --                         print(suggestion_i)
-- --                         print("suggestion")
--
--                         -- AnnotationComment
--                         local anno_comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
--                         anno_comment:setReference(spellout_anno)
--                         anno_comment:setKey("Spelling")
--                         anno_comment:setValue(model_name)
--                         anno_comment:addToIndexes()
-- --                         print("comment")
--
--                         --SuggestedAction into Anomaly
--                         spellout_anomaly:setSuggestions(counter_suggest, spellout_anno)
-- --                         print("Add anomaly")
--
--                         counter_suggest = counter_suggest + 1
-- --                         print(counter_suggest)
--                     end
                end
            end
        end
-- end

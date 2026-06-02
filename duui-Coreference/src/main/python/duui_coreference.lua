-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIutils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Coreference = luajava.bindClass("org.texttechnologylab.annotation.Coreference")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
--     print("start")
    local doc_text = inputCas:getDocumentText()
--     print(doc_text)
    local doc_lang = inputCas:getDocumentLanguage()
    local tokens = {}
    local begin_token = {}
    local end_token = {}
    local tokens_count = 1
    local tokens_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Token)):listIterator()
    while tokens_it:hasNext() do
        local token = tokens_it:next()
        tokens[tokens_count] = token:getCoveredText()
        begin_token[tokens_count] = token:getBegin()
        end_token[tokens_count] = token:getEnd()
        tokens_count = tokens_count + 1
    end
--     print("sentences")
--     print(tokens)
--     print(begin_token)
--     print(end_token)
    outputStream:write(json.encode({
        tokens = tokens,
        lang = doc_lang,
        begin_token = begin_token,
        end_token = end_token,
    }))
-- --     print("sendToPython")
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
--     print("begin_deserialize")

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["begin_resolve"] ~= nil then
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
        print(model_version)
        model_meta:setModelName(model_name)
        print(model_name)
        model_meta:setSource(source)
        print(source)
        model_meta:setLang(model_lang)
        print(model_lang)
        model_meta:addToIndexes()

        local meta = results["meta"]
--         print("meta")
        local begin = results["begin"]
        local end_token = results["end"]
        local begin_resolve = results["begin_resolve"]
        local end_resolve = results["end_resolve"]
        for index_i, begin_i in ipairs(begin) do
            local end_i = end_token[index_i]
            local begin_resolve_i = begin_resolve[index_i]
            local end_resolve_i = end_resolve[index_i]
            local coref_resolve = JCasUtil:selectAt(inputCas, Coreference, begin_resolve_i, end_resolve_i)
--             print(coref_resolve)
            if coref_resolve:size() == 0 then
                 coref_resolve = luajava.newInstance("org.texttechnologylab.annotation.Coreference", inputCas, begin_resolve_i, end_resolve_i)
            else
                coref_resolve = coref_resolve:iterator():next()
            end
--             print(coref_resolve)
            local coref_anno = luajava.newInstance("org.texttechnologylab.annotation.Coreference", inputCas, begin_i, end_i)
            coref_anno:setLink(coref_resolve)
            coref_anno:addToIndexes()
        end

--         local meta = results["meta"]
-- --         print("meta")
--         local begin_claims = results["begin_claims"]
-- --         print("begin_claims")
--         local end_claims = results["end_claims"]
-- --         print("end_claims")
--         local begin_facts = results["begin_facts"]
-- --         print("begin_facts")
--         local end_facts = results["end_facts"]
-- --         print("end_facts")
--         local consistency = results["consistency"]
-- --         print("consistency")
--         for index_i, cons in ipairs(consistency) do
-- --             print(cons)
--             local begin_claim_i = begin_claims[index_i]
-- --             print(begin_claim_i)
--             local end_claim_i = end_claims[index_i]
-- --             print(end_claim_i)
--             local begin_fact_i = begin_facts[index_i]
-- --             print(begin_fact_i)
--             local end_fact_i = end_facts[index_i]
-- --             print(end_fact_i)
--             local claim_i = util:selectAt(inputCas, claims, begin_claim_i, end_claim_i):iterator():next()
-- --             print(claim_i)
--             local fact_i  = util:selectAt(inputCas, facts, begin_fact_i, end_fact_i):iterator():next()
-- --             print(fact_i)
--             local factcheck_i = luajava.newInstance("org.texttechnologylab.annotation.FactChecking", inputCas)
-- --             print("FactCheck")
--             factcheck_i:setClaim(claim_i)
-- --             print("claim")
--             factcheck_i:setFact(fact_i)
-- --             print("fact")
--             factcheck_i:setConsistency(cons)
-- --             print("cons")
--             factcheck_i:setModel(model_meta)
-- --             print("setModel")
--             factcheck_i:addToIndexes()
-- --             print(factcheck_i)
--         end
    end
end

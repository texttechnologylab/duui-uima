-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
facts = luajava.bindClass("org.texttechnologylab.annotation.Fact")
claims = luajava.bindClass("org.texttechnologylab.annotation.Claim")
FactCheck = luajava.bindClass("org.texttechnologylab.annotation.FactChecking")

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
    local doc_lang = inputCas:getDocumentLanguage()
--     print(doc_text)
--     print(doc_lang)
    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    local all_facts = {}
    local all_claims = {}
    local sen_counter = 1
--     print("start")
    local claims_in = util:select(inputCas, claims):iterator()
    while claims_in:hasNext() do
        local claim = claims_in:next()
        local begin_claim = claim:getBegin()
        local end_claim = claim:getEnd()
        local claim_text = claim:getCoveredText()
--         print(claim_text)
--         print(begin_claim)
--         print(end_claim)
        all_claims[sen_counter] = {}
        all_claims[sen_counter]["begin"] = begin_claim
        all_claims[sen_counter]["end"] = end_claim
        all_claims[sen_counter]["text"] = claim_text
        all_claims[sen_counter]["facts"] = {}
        local facts_in = claim:getFacts():iterator()
        local fact_counter = 1
        while facts_in:hasNext() do
            local fact = facts_in:next()
            local begin_fact = fact:getBegin()
            local end_fact = fact:getEnd()
            local fact_text = fact:getCoveredText()
--             print("fact")
--             print(fact_text)
--             print(begin_fact)
--             print(end_fact)
            all_claims[sen_counter]["facts"][fact_counter] = {}
            all_claims[sen_counter]["facts"][fact_counter]["begin"] = begin_fact
            all_claims[sen_counter]["facts"][fact_counter]["end"] = end_fact
            all_claims[sen_counter]["facts"][fact_counter]["text"] = fact_text
            fact_counter = fact_counter + 1
        end
        sen_counter = sen_counter + 1
    end
    local fact_counter = 1
    local facts_in = util:select(inputCas, facts):iterator()
    while facts_in:hasNext() do
--         print("claims")
        local fact_now = facts_in:next()
--         print(fact)
        local begin_fact = fact_now:getBegin()
        local end_fact = fact_now:getEnd()
        local fact_text = fact_now:getCoveredText()
        all_facts[fact_counter] = {}
        all_facts[fact_counter]["begin"] = begin_fact
        all_facts[fact_counter]["end"] = end_fact
        all_facts[fact_counter]["text"] = fact_text
        all_facts[fact_counter]["claims"] = {}
        local claim_counter = 1
        local claims_in = fact_now:getClaims():iterator()
        while claims_in:hasNext() do
            local claim = claims_in:next()
            local begin_claim = claim:getBegin()
            local end_claim = claim:getEnd()
            local claim_text = claim:getCoveredText()
            all_facts[fact_counter]["claims"][claim_counter] = {}
            all_facts[fact_counter]["claims"][claim_counter]["begin"] = begin_claim
            all_facts[fact_counter]["claims"][claim_counter]["end"] = end_claim
            all_facts[fact_counter]["claims"][claim_counter]["text"] = claim_text
            claim_counter = claim_counter + 1
        end
        fact_counter = fact_counter + 1
    end
--     print("sentences")
    outputStream:write(json.encode({
        claims_all = all_claims,
        facts_all = all_facts
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

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["consistency"] ~= nil then
        print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]
        print("meta")
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        print("setMetaData")
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
        local begin_claims = results["begin_claims"]
--         print("begin_claims")
        local end_claims = results["end_claims"]
--         print("end_claims")
        local begin_facts = results["begin_facts"]
--         print("begin_facts")
        local end_facts = results["end_facts"]
--         print("end_facts")
        local consistency = results["consistency"]
--         print("consistency")
        for index_i, cons in ipairs(consistency) do
--             print(cons)
            local begin_claim_i = begin_claims[index_i]
--             print(begin_claim_i)
            local end_claim_i = end_claims[index_i]
--             print(end_claim_i)
            local begin_fact_i = begin_facts[index_i]
--             print(begin_fact_i)
            local end_fact_i = end_facts[index_i]
--             print(end_fact_i)
            local claim_i = util:selectAt(inputCas, claims, begin_claim_i, end_claim_i):iterator():next()
--             print(claim_i)
            local fact_i  = util:selectAt(inputCas, facts, begin_fact_i, end_fact_i):iterator():next()
--             print(fact_i)
            local factcheck_i = luajava.newInstance("org.texttechnologylab.annotation.FactChecking", inputCas)
--             print("FactCheck")
            factcheck_i:setClaim(claim_i)
--             print("claim")
            factcheck_i:setFact(fact_i)
--             print("fact")
            factcheck_i:setConsistency(cons)
--             print("cons")
            factcheck_i:setModel(model_meta)
--             print("setModel")
            factcheck_i:addToIndexes()
--             print(factcheck_i)
        end
    end
end

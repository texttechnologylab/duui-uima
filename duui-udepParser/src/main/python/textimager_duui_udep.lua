-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
-- POS = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS")
-- Dep = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
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
    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    local sentences_cas = {}
    local token_cas = {}
    local sen_counter = 1
    local sents = util:select(inputCas, Sentence):iterator()
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
            token_type = token:getType():getName()
            if token_type == "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token" then
                token_cas[sen_counter][token_counter] = {}
                token_cas[sen_counter][token_counter]["begin"] = token_begin
                token_cas[sen_counter][token_counter]["end"] = token_end
                token_cas[sen_counter][token_counter]["text"] = token:getCoveredText()
--                 local pos = util:selectCovered(POS, token):iterator()
--                 if pos:hasNext() then
--                     pos_ = pos:next()
--                     token_cas[sen_counter][token_counter]["pos"] = pos_:getCoarseValue()
--                     token_cas[sen_counter][token_counter]["tag"] = pos_:getPosValue()
--                 end
--                 local dep = util:selectCovered(Dep, token):iterator()
--                 if dep:hasNext() then
--                     dep_ = dep:next()
--                     token_cas[sen_counter][token_counter]["dep"] = dep_:getDependencyType()
--                 else
--                     token_cas[sen_counter][token_counter]["dep"] = "X"
--                 end
                token_counter = token_counter + 1
            end
        end
        sen_counter = sen_counter + 1
    end

    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        tokens = token_cas
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
--     print('enter deserializing')
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
--     Get Token results
    local udeps = results["udeps"]
    local tokens = results["tokens"]
    cas_udeps = {}

    token_lookup = {}
    local token_selection = util:select(inputCas, Token):iterator()

    while token_selection:hasNext() do
        local token = token_selection:next()
        local begin_t = token:getBegin()
        local end_t = token:getEnd()
        token_lookup[begin_t] = {}
        token_lookup[begin_t][end_t] = token
    end

    for i, udep in ipairs(udeps) do
        if udep["write"] then
            local cas_udep = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", inputCas)
            b = udep["begin"]
            e = udep["end"]
            rel = udep["type"]
            cas_udep:setBegin(b)
            cas_udep:setEnd(e)
            dependent_ind = udep["dependent_ind"]
            governor_ind = i - dependent_ind + udep["governor_ind"]
            governor_b = tokens[governor_ind]["begin"]
            governor_e = tokens[governor_ind]["end"]
            dependent_token = token_lookup[b][e]
            governor_token = token_lookup[governor_b][governor_e]

            if governor_token ~= nil then
--             if governor_token:hasNext() then
--                 gt = governor_token:next()
                cas_udep:setGovernor(governor_token)
            end

            if dependent_token ~= nil then
--             if dependent_token:hasNext() then
--                 dt = dependent_token:next()
                cas_udep:setDependent(dependent_token)
            else
                print('NIL', b, e, governor_b, governor_e)
            end

--             if dt ~= nil and gt ~= nil then
--                 dt:setParent(gt)
--             end
            cas_udep:setDependencyType(rel)
            cas_udep:setFlavor("udep")
            cas_udep:addToIndexes()
--             cas_udeps[b] = {}
--             cas_udeps[b][e] = cas_udep

--             local cas_udep_p = luajava.newInstance("org.texttechnologylab.annotation.AnnotationPerspective", inputCas)
--             cas_udep_p:setName("udep")
--             cas_udep_p:setReference(cas_udep)
--             cas_udep_p:addToIndexes()


        end
    end
end

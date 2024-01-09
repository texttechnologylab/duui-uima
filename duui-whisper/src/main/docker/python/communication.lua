-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local audioBase64 = inputCas:getView("audio"):getSofaDataString()

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        audio = audioBase64
    }))
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

    -- Sentences
    if results["audio_token"] ~= null then
        for i, sent in ipairs(results["audio_token"]) do
            local audioToken = luajava.newInstance("org.texttechnologylab.annotation.type.AudioToken", inputCas)
            audioToken:setBegin(sent["begin"])
            audioToken:setEnd(sent["end"])
            audioToken:addToIndexes()
        end
    end

    -- Token (+pos+lemma+morph)
    local token_list = {}
    local token_counter = 1
    if results["token"] ~= null then
        for i, tok in ipairs(results["token"]) do
            -- POS
            local pos = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", inputCas)
            pos:setBegin(tok["begin"])
            pos:setEnd(tok["end"])
            pos:setPosValue(tok["pos"]["PosValue"])
            pos:setCoarseValue(tok["pos"]["coarseValue"])
            pos:addToIndexes()

            -- Lemma
            local lemma = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma", inputCas)
            lemma:setBegin(tok["begin"])
            lemma:setEnd(tok["end"])
            lemma:setValue(tok["lemma"]["value"])
            lemma:addToIndexes()

            -- MorphologicalFeatures
            local morph = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", inputCas)
            morph:setBegin(tok["begin"])
            morph:setEnd(tok["end"])
            if tok["morph"]["gender"] ~= null then
                morph:setGender(tok["morph"]["gender"])
            end
            if tok["morph"]["case"] ~= null then
                morph:setCase(tok["morph"]["case"])
            end
            if tok["morph"]["number"] ~= null then
                morph:setNumber(tok["morph"]["number"])
            end
            if tok["morph"]["degree"] ~= null then
                morph:setDegree(tok["morph"]["degree"])
            end
            if tok["morph"]["verbForm"] ~= null then
                morph:setVerbForm(tok["morph"]["verbForm"])
            end
            if tok["morph"]["tense"] ~= null then
                morph:setTense(tok["morph"]["tense"])
            end
            if tok["morph"]["mood"] ~= null then
                morph:setMood(tok["morph"]["mood"])
            end
            if tok["morph"]["voice"] ~= null then
                morph:setVoice(tok["morph"]["voice"])
            end
            if tok["morph"]["definiteness"] ~= null then
                morph:setDefiniteness(tok["morph"]["definiteness"])
            end
            if tok["morph"]["value"] ~= null then
                morph:setValue(tok["morph"]["value"])
            end
            if tok["morph"]["person"] ~= null then
                morph:setPerson(tok["morph"]["person"])
            end
            if tok["morph"]["aspect"] ~= null then
                morph:setAspect(tok["morph"]["aspect"])
            end
            if tok["morph"]["animacy"] ~= null then
                morph:setAnimacy(tok["morph"]["animacy"])
            end
            if tok["morph"]["negative"] ~= null then
                morph:setNegative(tok["morph"]["negative"])
            end
            if tok["morph"]["numType"] ~= null then
                morph:setNumType(tok["morph"]["numType"])
            end
            if tok["morph"]["possessive"] ~= null then
                morph:setPossessive(tok["morph"]["possessive"])
            end
            if tok["morph"]["pronType"] ~= null then
                morph:setPronType(tok["morph"]["pronType"])
            end
            if tok["morph"]["reflex"] ~= null then
                morph:setReflex(tok["morph"]["reflex"])
            end
            if tok["morph"]["transitivity"] ~= null then
                morph:setTransitivity(tok["morph"]["transitivity"])
            end
            morph:addToIndexes()

            -- Token
            local token_obj = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", inputCas)
            token_obj:setBegin(tok["begin"])
            token_obj:setEnd(tok["end"])
            token_obj:setLemma(lemma)
            token_obj:setPos(pos)
            token_obj:setMorph(morph)
            token_obj:addToIndexes()

            token_list[token_counter] = token_obj
            token_counter = token_counter + 1
        end
    end

    -- deps
    if results["deps"] ~= null then
        for i, dep in ipairs(results["deps"]) do
            local dependency = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", inputCas)
            dependency:setBegin(dep["begin"])
            dependency:setEnd(dep["end"])
            dependency:setFlavor(dep["flavor"])
            dependency:setDependencyType(dep["DependencyType"])
            dependency:setGovernor(token_list[dep["Governor"] + 1])
            dependency:setDependent(token_list[dep["Dependent"] + 1])
            dependency:addToIndexes()
        end
    end

    -- ners
    if results["ners"] ~= null then
        for i, ner in ipairs(results["ners"]) do
            local entity = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", inputCas)
            entity:setBegin(ner["begin"])
            entity:setEnd(ner["end"])
            entity:setValue(ner["value"])
            entity:addToIndexes()
        end
    end
end

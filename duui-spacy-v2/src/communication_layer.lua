-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")

REQUEST_BATCH_SIZE = 1024

---Get batches of sentences from the JCas using a coroutine.
---@param jCas any JCas (view) containing sentence annotations which are to be processed
---@param batch_size integer size of each batch sent to the component
function get_batches(jCas, batch_size)
    local batch = {}
    local iterator = JCasUtil:select(jCas, Sentence):iterator()
    while iterator:hasNext() do
        sentence = iterator:next()
        batch[#batch + 1] = {
            text = sentence:getCoveredText(),
            offset = sentence:getBegin(),
        }
        if #batch == batch_size then
            coroutine.yield(batch)
            batch = {}
        end
    end

    if #batch > 0 then
        coroutine.yield(batch)
    end
end

---Iterate over batches of sentences from the JCas.
---@param jCas any JCas to process
---@param batch_size integer size of each batch
---@return fun(): table an iterator over batches to process
function batched(jCas, batch_size)
    local co = coroutine.create(function() get_batches(jCas, batch_size) end)
    return function()
        local _, batch = coroutine.resume(co)
        return batch
    end
end

---Process the sentences in the given JCas in small batches.
---@param source any JCas (view) to process
---@param handler any DuuiHttpRequestHandler with a connection to the running component
---@param parameters table optional parameters
---@param target any JCas (view) to write the results to (optional)
function process(source, handler, parameters, target)
    parameters = parameters or {}
    local config = {
        spacy_language = source:getDocumentLanguage(),
        spacy_model_size = parameters.spacy_model_size or "lg",
        spacy_batch_size = parameters.spacy_batch_size or 32,
    }
    local batch_size = parameters.request_batch_size or REQUEST_BATCH_SIZE

    for batch in batched(source, batch_size) do
        process_response(
            target or source,
            handler:process(
                json.encode({
                    sentences = batch,
                    config = config,
                })
            )
        )
    end
end

---Process the response from the component.
---@param jCas any JCas
---@param response any DuuiHttpRequestHandler.Response{int statusCode, byte[]? body}
function process_response(jCas, response)
    if response:statusCode() ~= 200 then
        error("Error " .. response:statusCode() .. " in communication with component: " .. response:body())
    end

    local results = json.decode(response:bodyUtf8())

    local tokens = {}

    for i, token in ipairs(results.tokens) do
        local token_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", jCas)
        token_anno:setBegin(token["begin"])
        token_anno:setEnd(token["end"])
        token_anno:addToIndexes()

        tokens[i] = token_anno

        if token.lemma ~= nil then
            local lemma_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma", jCas)
            lemma_anno:setBegin(token["begin"])
            lemma_anno:setEnd(token["end"])
            lemma_anno:setValue(token["lemma"])
            token_anno:setLemma(lemma_anno)
            lemma_anno:addToIndexes()
        end

        if token.pos_value ~= nil then
            local pos_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", jCas)
            pos_anno:setBegin(token["begin"])
            pos_anno:setEnd(token["end"])
            pos_anno:setPosValue(token["pos_value"])
            pos_anno:setCoarseValue(token["pos_coarse"])
            token_anno:setPos(pos_anno)
            pos_anno:addToIndexes()
        end

        if token.morph_value ~= nil and token.morph_value ~= "" then
            local morph_anno = luajava.newInstance(
                "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", jCas)
            morph_anno:setBegin(token["begin"])
            morph_anno:setEnd(token["end"])
            morph_anno:setValue(token["morph_value"])
            token_anno:setMorph(morph_anno)
            morph_anno:addToIndexes()

            -- Add detailed infos, if available
            if token["morph_features"]["Gender"] ~= nil then
                morph_anno:setGender(token["morph_features"]["Gender"])
            end
            if token["morph_features"]["Number"] ~= nil then
                morph_anno:setNumber(token["morph_features"]["Number"])
            end
            if token["morph_features"]["Case"] ~= nil then
                morph_anno:setCase(token["morph_features"]["Case"])
            end
            if token["morph_features"]["Degree"] ~= nil then
                morph_anno:setDegree(token["morph_features"]["Degree"])
            end
            if token["morph_features"]["VerbForm"] ~= nil then
                morph_anno:setVerbForm(token["morph_features"]["VerbForm"])
            end
            if token["morph_features"]["Tense"] ~= nil then
                morph_anno:setTense(token["morph_features"]["Tense"])
            end
            if token["morph_features"]["Mood"] ~= nil then
                morph_anno:setMood(token["morph_features"]["Mood"])
            end
            if token["morph_features"]["Voice"] ~= nil then
                morph_anno:setVoice(token["morph_features"]["Voice"])
            end
            if token["morph_features"]["Definiteness"] ~= nil then
                morph_anno:setDefiniteness(token["morph_features"]["Definiteness"])
            end
            if token["morph_features"]["Person"] ~= nil then
                morph_anno:setPerson(token["morph_features"]["Person"])
            end
            if token["morph_features"]["Aspect"] ~= nil then
                morph_anno:setAspect(token["morph_features"]["Aspect"])
            end
            if token["morph_features"]["Animacy"] ~= nil then
                morph_anno:setAnimacy(token["morph_features"]["Animacy"])
            end
            if token["morph_features"]["Negative"] ~= nil then
                morph_anno:setNegative(token["morph_features"]["Negative"])
            end
            if token["morph_features"]["NumType"] ~= nil then
                morph_anno:setNumType(token["morph_features"]["NumType"])
            end
            if token["morph_features"]["Possessive"] ~= nil then
                morph_anno:setPossessive(token["morph_features"]["Possessive"])
            end
            if token["morph_features"]["PronType"] ~= nil then
                morph_anno:setPronType(token["morph_features"]["PronType"])
            end
            if token["morph_features"]["Reflex"] ~= nil then
                morph_anno:setReflex(token["morph_features"]["Reflex"])
            end
            if token["morph_features"]["Transitivity"] ~= nil then
                morph_anno:setTransitivity(token["morph_features"]["Transitivity"])
            end
        end
    end


    for _, dep in ipairs(results.dependencies) do
        local dep_anno
        local dep_type = string.upper(dep["dependency_type"])
        if dep_type == "ROOT" then
            dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT", jCas)
            dep_anno:setDependencyType("--")
        else
            if dep_type == "ABBREV" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ABBREV", jCas)
            elseif dep_type == "ACOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ACOMP", jCas)
            elseif dep_type == "ADVCL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVCL", jCas)
            elseif dep_type == "ADVMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVMOD", jCas)
            elseif dep_type == "AGENT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AGENT", jCas)
            elseif dep_type == "AMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AMOD", jCas)
            elseif dep_type == "APPOS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.APPOS", jCas)
            elseif dep_type == "ATTR" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ATTR", jCas)
            elseif dep_type == "AUX0" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUX0", jCas)
            elseif dep_type == "AUXPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUXPASS", jCas)
            elseif dep_type == "CC" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CC", jCas)
            elseif dep_type == "CCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CCOMP", jCas)
            elseif dep_type == "COMPLM" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COMPLM", jCas)
            elseif dep_type == "CONJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ", jCas)
            elseif dep_type == "CONJP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJP", jCas)
            elseif dep_type == "CONJ_YET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ_YET", jCas)
            elseif dep_type == "COP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COP", jCas)
            elseif dep_type == "CSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJ", jCas)
            elseif dep_type == "CSUBJPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJPASS", jCas)
            elseif dep_type == "DEP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DEP", jCas)
            elseif dep_type == "DET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DET", jCas)
            elseif dep_type == "DOBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DOBJ", jCas)
            elseif dep_type == "EXPL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.EXPL", jCas)
            elseif dep_type == "INFMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.INFMOD", jCas)
            elseif dep_type == "IOBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.IOBJ", jCas)
            elseif dep_type == "MARK" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MARK", jCas)
            elseif dep_type == "MEASURE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MEASURE", jCas)
            elseif dep_type == "MWE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MWE", jCas)
            elseif dep_type == "NEG" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NEG", jCas)
            elseif dep_type == "NN" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NN", jCas)
            elseif dep_type == "NPADVMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NPADVMOD", jCas)
            elseif dep_type == "NSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJ", jCas)
            elseif dep_type == "NSUBJPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJPASS", jCas)
            elseif dep_type == "NUM" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUM", jCas)
            elseif dep_type == "NUMBER" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUMBER", jCas)
            elseif dep_type == "PARATAXIS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARATAXIS", jCas)
            elseif dep_type == "PARTMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARTMOD", jCas)
            elseif dep_type == "PCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PCOMP", jCas)
            elseif dep_type == "POBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POBJ", jCas)
            elseif dep_type == "POSS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSS", jCas)
            elseif dep_type == "POSSESSIVE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSSESSIVE",
                jCas)
            elseif dep_type == "PRECONJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRECONJ", jCas)
            elseif dep_type == "PRED" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRED", jCas)
            elseif dep_type == "PREDET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREDET", jCas)
            elseif dep_type == "PREP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREP", jCas)
            elseif dep_type == "PREPC" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREPC", jCas)
            elseif dep_type == "PRT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRT", jCas)
            elseif dep_type == "PUNCT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PUNCT", jCas)
            elseif dep_type == "PURPCL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PURPCL", jCas)
            elseif dep_type == "QUANTMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.QUANTMOD", jCas)
            elseif dep_type == "RCMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.RCMOD", jCas)
            elseif dep_type == "REF" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REF", jCas)
            elseif dep_type == "REL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REL", jCas)
            elseif dep_type == "ROOT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT", jCas)
            elseif dep_type == "TMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.TMOD", jCas)
            elseif dep_type == "XSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XSUBJ", jCas)
            elseif dep_type == "XCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XCOMP", jCas)
            else
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency",
                    jCas)
            end
            dep_anno:setDependencyType(dep["type"])
        end

        dep_anno:setBegin(dep["begin"])
        dep_anno:setEnd(dep["end"])
        dep_anno:setFlavor(dep["flavor"])

        local governor = tokens[dep["governor_index"] + 1]
        if governor ~= nil then
            dep_anno:setGovernor(governor)
        end

        local dependent = tokens[dep["dependent_index"] + 1]
        if dependent ~= nil then
            dep_anno:setDependent(dependent)
        end

        if governor ~= nil and dependent ~= nil then
            dependent:setParent(governor)
        end
    end

    for _, entity in ipairs(results.entities) do
        local entity_anno

        local entity_value = entity["value"]
        local ENTITY_VALUE = string.upper(entity_value)
        if entity_value == "Organization" or ENTITY_VALUE == "ORG" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Organization", jCas)
        elseif entity_value == "Person" or ENTITY_VALUE == "PER" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Person", jCas)
        elseif entity_value == "Location" or ENTITY_VALUE == "LOC" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location", jCas)
        else
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", jCas)
        end

        entity_anno:setBegin(entity["begin"])
        entity_anno:setEnd(entity["end"])
        entity_anno:setValue(entity_value)
        entity_anno:addToIndexes()
    end
end

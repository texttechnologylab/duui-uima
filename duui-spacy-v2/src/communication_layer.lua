-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")

REQUEST_BATCH_SIZE = 1024

-- TODO
-- Inputs:
--  - jCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: Table/Dictonary of parameters that should be used to configure the annotator
---comment
---@param jCas any
---@param handler any
---@param parameters table
function process(jCas, handler, parameters)

    local config = {
        spacy_language = jCas:getDocumentLanguage(),
        spacy_model_size = parameters.spacy_model_size or "lg",
        spacy_batch_size = parameters.spacy_batch_size or 32,
    }

    local i, batch, sentence, batchIndex = 0, {}
    local iterator = JCasUtil:select(jCas, Sentence):iterator()
    while iterator:hasNext() do
        sentence = iterator:next()
        batchIndex = i % REQUEST_BATCH_SIZE + 1
        batch[batchIndex] = {
            text = sentence:getCoveredText(),
            offset = sentence:getBegin(),
        }

        if batchIndex == REQUEST_BATCH_SIZE then
            local response = handler:process(json.encode({
                sentences = batch,
                config = config,
            }))
            local results = json.decode(luajava.newInstance("java.lang.String", response:readAllBytes(),
                StandardCharsets.UTF_8))
            process_results(jCas, results)
            batch = {}
        end

        i = i + 1
    end
    if #batch > 0 then
        local response = handler:process(json.encode({
            sentences = batch,
            config = config,
        }))
        local results = json.decode(luajava.newInstance("java.lang.String", response:readAllBytes(),
            StandardCharsets.UTF_8))
        process_results(jCas, results)
    end
end

---comment
---@param jCas any
---@param results table
function process_results(jCas, results)
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
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ABBREV")
            elseif dep["type"] == "ACOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ACOMP")
            elseif dep["type"] == "ADVCL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVCL")
            elseif dep["type"] == "ADVMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVMOD")
            elseif dep["type"] == "AGENT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AGENT")
            elseif dep["type"] == "AMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AMOD")
            elseif dep["type"] == "APPOS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.APPOS")
            elseif dep["type"] == "ATTR" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ATTR")
            elseif dep["type"] == "AUX0" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUX0")
            elseif dep["type"] == "AUXPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUXPASS")
            elseif dep["type"] == "CC" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CC")
            elseif dep["type"] == "CCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CCOMP")
            elseif dep["type"] == "COMPLM" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COMPLM")
            elseif dep["type"] == "CONJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ")
            elseif dep["type"] == "CONJP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJP")
            elseif dep["type"] == "CONJ_YET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ_YET")
            elseif dep["type"] == "COP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COP")
            elseif dep["type"] == "CSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJ")
            elseif dep["type"] == "CSUBJPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJPASS")
            elseif dep["type"] == "DEP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DEP")
            elseif dep["type"] == "DET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DET")
            elseif dep["type"] == "DOBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DOBJ")
            elseif dep["type"] == "EXPL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.EXPL")
            elseif dep["type"] == "INFMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.INFMOD")
            elseif dep["type"] == "IOBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.IOBJ")
            elseif dep["type"] == "MARK" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MARK")
            elseif dep["type"] == "MEASURE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MEASURE")
            elseif dep["type"] == "MWE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MWE")
            elseif dep["type"] == "NEG" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NEG")
            elseif dep["type"] == "NN" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NN")
            elseif dep["type"] == "NPADVMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NPADVMOD")
            elseif dep["type"] == "NSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJ")
            elseif dep["type"] == "NSUBJPASS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJPASS")
            elseif dep["type"] == "NUM" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUM")
            elseif dep["type"] == "NUMBER" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUMBER")
            elseif dep["type"] == "PARATAXIS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARATAXIS")
            elseif dep["type"] == "PARTMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARTMOD")
            elseif dep["type"] == "PCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PCOMP")
            elseif dep["type"] == "POBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POBJ")
            elseif dep["type"] == "POSS" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSS")
            elseif dep["type"] == "POSSESSIVE" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSSESSIVE")
            elseif dep["type"] == "PRECONJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRECONJ")
            elseif dep["type"] == "PRED" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRED")
            elseif dep["type"] == "PREDET" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREDET")
            elseif dep["type"] == "PREP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREP")
            elseif dep["type"] == "PREPC" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREPC")
            elseif dep["type"] == "PRT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRT")
            elseif dep["type"] == "PUNCT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PUNCT")
            elseif dep["type"] == "PURPCL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PURPCL")
            elseif dep["type"] == "QUANTMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.QUANTMOD")
            elseif dep["type"] == "RCMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.RCMOD")
            elseif dep["type"] == "REF" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REF")
            elseif dep["type"] == "REL" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REL")
            elseif dep["type"] == "ROOT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT")
            elseif dep["type"] == "TMOD" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.TMOD")
            elseif dep["type"] == "XSUBJ" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XSUBJ")
            elseif dep["type"] == "XCOMP" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XCOMP")
            else
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency",
                    jCas)
            end
            dep_anno:setDependencyType(dep["type"])
        end

        dep_anno:setBegin(dep["begin"])
        dep_anno:setEnd(dep["end"])
        dep_anno:setFlavor(dep["flavor"])

        governor = tokens[dep["governor_index"] + 1]
        if governor ~= nil then
            dep_anno:setGovernor(governor)
        end

        dependent = tokens[dep["dependent_index"] + 1]
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

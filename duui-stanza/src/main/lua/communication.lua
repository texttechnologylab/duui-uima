StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")

SENTENCE_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
STANZA_SENTENCE_TYPE_NAME = "org.texttechnologylab.uima.type.stanza.StanzaSentence"
STANZA_TOKEN_TYPE_NAME = "org.texttechnologylab.uima.type.stanza.StanzaToken"

-- The given spans of exactly this type: subtypes (e.g. the StanzaSentence / StanzaToken of an earlier run, SpacyToken) are
-- layers of other tools and not the input
function exactSpans(inputCas, class, type_name)
    local spans = {}
    -- by index: LuaJ cannot call the methods of the JDK's private iterator classes (Java module system)
    local list = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, class))
    for i = 0, list:size() - 1 do
        local annotation = list:get(i)
        if annotation:getType():getName() == type_name then
            spans[#spans + 1] = {
                begin = annotation:getBegin(),
                ['end'] = annotation:getEnd(),
            }
        end
    end
    return spans
end

function serialize(inputCas, outputStream, parameters)
    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        len = DUUIUtils:getDocumentTextLength(inputCas),
        lang = inputCas:getDocumentLanguage(),
        sentences = exactSpans(inputCas, Sentence, SENTENCE_TYPE_NAME),
        tokens = exactSpans(inputCas, Token, TOKEN_TYPE_NAME),
    }))
end

-- The Stanza types are created through the CAS API, so that the client needs no JCas classes for them (the type comes from
-- the component's type system)
function createAnnotation(cas, type_name, begin, end_)
    local annotation_type = cas:getTypeSystem():getType(type_name)
    if annotation_type == nil then
        error("Type is missing from the type system: " .. type_name)
    end
    local annotation = cas:createAnnotation(annotation_type, begin, end_)
    cas:addFsToIndexes(annotation)
    return annotation
end

function addMeta(inputCas, reference, meta)
    local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
    meta_anno:setReference(reference)
    meta_anno:setName(meta["name"])
    meta_anno:setVersion(meta["version"])
    meta_anno:setModelName(meta["modelName"])
    meta_anno:setModelVersion(meta["modelVersion"])
    meta_anno:addToIndexes()
end

-- UD feature names -> setters of the DKPro MorphologicalFeatures; the complete feature string (with e.g. Number[psor]) is the
-- value
MORPH_SETTERS = {
    Gender = "setGender", Number = "setNumber", Case = "setCase", Degree = "setDegree", VerbForm = "setVerbForm",
    Tense = "setTense", Mood = "setMood", Voice = "setVoice", Definite = "setDefiniteness", Person = "setPerson",
    Aspect = "setAspect", Animacy = "setAnimacy", Polarity = "setNegative", NumType = "setNumType", Poss = "setPossessive",
    PronType = "setPronType", Reflex = "setReflex",
}

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    local cas = inputCas:getCas()

    local modification_meta = results["modification_meta"]
    local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
    modification_anno:setUser(modification_meta["user"])
    modification_anno:setTimestamp(modification_meta["timestamp"])
    modification_anno:setComment(modification_meta["comment"])
    modification_anno:addToIndexes()

    local meta = results["meta"]
    for _, sentence in ipairs(results["sentences"]) do
        local sentence_anno = createAnnotation(cas, STANZA_SENTENCE_TYPE_NAME, sentence["begin"], sentence["end"])
        addMeta(inputCas, sentence_anno, meta)

        -- one StanzaToken per word; the words of a split contraction share the span of their token and carry the word
        -- as TokenForm
        local tokens = {}
        for i, word in ipairs(sentence["words"]) do
            local token_anno = createAnnotation(cas, STANZA_TOKEN_TYPE_NAME, word["begin"], word["end"])
            tokens[i] = token_anno
            addMeta(inputCas, token_anno, meta)

            if word["form"] ~= nil then
                local form_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.TokenForm", inputCas)
                form_anno:setBegin(word["begin"])
                form_anno:setEnd(word["end"])
                form_anno:setValue(word["form"])
                form_anno:addToIndexes()
                token_anno:setForm(form_anno)
            end

            local pos_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", inputCas)
            pos_anno:setBegin(word["begin"])
            pos_anno:setEnd(word["end"])
            pos_anno:setPosValue(word["xpos"])
            pos_anno:setCoarseValue(word["upos"])
            pos_anno:addToIndexes()
            token_anno:setPos(pos_anno)
            addMeta(inputCas, pos_anno, meta)

            if word["lemma"] ~= nil then
                local lemma_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma", inputCas)
                lemma_anno:setBegin(word["begin"])
                lemma_anno:setEnd(word["end"])
                lemma_anno:setValue(word["lemma"])
                lemma_anno:addToIndexes()
                token_anno:setLemma(lemma_anno)
                addMeta(inputCas, lemma_anno, meta)
            end

            if word["feats"] ~= nil then
                local morph_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", inputCas)
                morph_anno:setBegin(word["begin"])
                morph_anno:setEnd(word["end"])
                morph_anno:setValue(word["feats"])
                for feature in string.gmatch(word["feats"], "[^|]+") do
                    local key, value = string.match(feature, "^([^=]+)=(.*)$")
                    local setter = key ~= nil and MORPH_SETTERS[key] or nil
                    if setter ~= nil then
                        morph_anno[setter](morph_anno, value)
                    end
                end
                morph_anno:addToIndexes()
                token_anno:setMorph(morph_anno)
                addMeta(inputCas, morph_anno, meta)
            end
        end

        -- UD dependencies within the sentence; the root as DKPro ROOT with type "--" and itself as governor (as duui-spacy)
        for i, word in ipairs(sentence["words"]) do
            local dependent = tokens[i]
            local dep_anno
            local governor
            if word["head"] == nil then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT", inputCas)
                dep_anno:setDependencyType("--")
                governor = dependent
            else
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", inputCas)
                dep_anno:setDependencyType(word["deprel"])
                governor = tokens[word["head"] + 1]
                dependent:setParent(governor)
            end
            dep_anno:setBegin(word["begin"])
            dep_anno:setEnd(word["end"])
            dep_anno:setFlavor("basic")
            dep_anno:setGovernor(governor)
            dep_anno:setDependent(dependent)
            dep_anno:addToIndexes()
            addMeta(inputCas, dep_anno, meta)
        end
    end
end

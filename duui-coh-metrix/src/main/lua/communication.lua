StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
Double = luajava.bindClass("java.lang.Double")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
-- Coh-Metrix reads Universal Dependencies (README, "Prerequisites and preprocessing"): the DKPro types an upstream UD
-- parser writes (Token, POS, Lemma, MorphologicalFeatures, Dependency), e.g. duui-stanza; spaCy's TIGER / ClearNLP labels
-- are mapped to UD in duui.py as best effort (README, "Best effort without UD")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Paragraph = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph")
Dependency = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency")
TOKEN_TYPE_NAME = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
SPACY_TOKEN_TYPE_NAME = "org.texttechnologylab.uima.type.spacy.SpacyToken"
STANZA_TOKEN_TYPE_NAME = "org.texttechnologylab.uima.type.stanza.StanzaToken"
STANZA_SENTENCE_TYPE_NAME = "org.texttechnologylab.uima.type.stanza.StanzaSentence"
-- The token layer that is read, in this order: the words of duui-stanza (UD), the tokens of duui-spacy, plain DKPro Tokens.
-- A CAS can hold several layers over the same text (e.g. SoMaJo tokens, StanzaTokens and the SpacyTokens of the vectors);
-- reading more than one would count every word twice.
TOKEN_LAYERS = {STANZA_TOKEN_TYPE_NAME, SPACY_TOKEN_TYPE_NAME, TOKEN_TYPE_NAME}
-- optional: only needed to read the spaCy model of SpacyToken vectors
local spacy_meta_ok, SpacyAnnotatorMetaData = pcall(
    luajava.bindClass, "org.texttechnologylab.annotation.SpacyAnnotatorMetaData")
-- optional: token vectors as WordEmbedding annotations (type systems without it
-- only have SpacyToken.vector)
local word_embedding_ok, WordEmbedding = pcall(
    luajava.bindClass, "org.texttechnologylab.uima.type.WordEmbedding")

COHMETRIX_TYPE_PREFIX = "org.texttechnologylab.uima.type.cohmetrix."

-- The concrete UIMA type is the public label: the Coh-Metrix 3 label, or
-- <NAME>_ext for project-only indices.
function resolveIndexTypeName(index)
    local label = index["label_v3"]

    if label == nil or label == "" then
        error("Cannot resolve Coh-Metrix UIMA type: label_v3 is missing")
    end

    -- All generated type names are based on labels and must be valid UIMA
    -- short names. Reject malformed labels instead of silently writing the
    -- annotation as the generic Index type.
    if string.match(label, "^[A-Za-z_][A-Za-z0-9_]*$") == nil then
        error("Invalid Coh-Metrix UIMA type label: " .. tostring(label))
    end

    return COHMETRIX_TYPE_PREFIX .. label
end

function requireFeature(index_type, feature_name)
    local feature = index_type:getFeatureByBaseName(feature_name)
    if feature == nil then
        error(
            "Coh-Metrix UIMA type "
            .. index_type:getName()
            .. " does not provide inherited feature "
            .. feature_name
        )
    end
    return feature
end

function setOptionalStringFeature(feature_structure, feature, value)
    if value ~= nil then
        feature_structure:setStringValue(feature, value)
    end
end

-- The type names of the annotations of a class in the CAS (subtypes included). By index: LuaJ cannot call the methods of the
-- JDK's private iterator classes (Java module system) without --add-opens.
function typeNames(inputCas, class)
    local names = {}
    local list = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, class))
    for i = 0, list:size() - 1 do
        names[list:get(i):getType():getName()] = true
    end
    return names
end

-- The token layer to read (see TOKEN_LAYERS); nil: every token (no known layer, e.g. only another tool's subtype)
function tokenLayer(inputCas)
    local names = typeNames(inputCas, Token)
    for _, name in ipairs(TOKEN_LAYERS) do
        if names[name] then
            return name
        end
    end
    return nil
end

-- The sentence layer to read: the StanzaSentences if there are any (a complete layer of duui-stanza, in which a sentence too
-- long for the parser is split), else every Sentence
function sentenceLayer(inputCas)
    if typeNames(inputCas, Sentence)[STANZA_SENTENCE_TYPE_NAME] then
        return STANZA_SENTENCE_TYPE_NAME
    end
    return nil
end

-- The annotations of a class covered by an annotation, only of the given type name (nil: all), in CAS order
function coveredOfType(class, cover, type_name)
    local all = JCasUtil:selectCovered(class, cover)
    local out = luajava.newInstance("java.util.ArrayList")
    for i = 0, all:size() - 1 do
        local annotation = all:get(i)
        if type_name == nil or annotation:getType():getName() == type_name then
            out:add(annotation)
        end
    end
    return out
end

-- The tokens of a sentence: those of the token layer
function sentenceTokens(sentence, token_layer)
    return coveredOfType(Token, sentence, token_layer)
end

-- The word of a token: the form of a split contraction (UD: German "im" = "in" + "dem",
-- two Tokens over the span of "im", each with its TokenForm), else the covered text.
function tokenWord(token)
    local form = token:getForm()
    if form ~= nil and form:getValue() ~= nil and form:getValue() ~= "" then
        return form:getValue()
    end
    return token:getCoveredText()
end

-- A UD feature from the MorphologicalFeatures: the DKPro feature if set, else from the
-- feature string ("Mood=Sub|Number=Plur|..."). Number[psor] is only in the string.
function morphFeature(morph, value, key)
    if value ~= nil and value ~= "" then
        return value
    end
    if morph == nil then
        return nil
    end
    local features = morph:getValue()
    if features == nil then
        return nil
    end
    local pattern = "^" .. key:gsub("%[", "%%["):gsub("%]", "%%]") .. "=(.*)$"
    for feature in string.gmatch(features, "[^|]+") do
        local match = string.match(feature, pattern)
        if match ~= nil then
            return match
        end
    end
    return nil
end

function sameAnnotation(a, b)
    return a ~= nil and b ~= nil and (a == b or a:equals(b))
end

-- The spaCy model of SpacyToken vectors (e.g. de_core_news_lg): the SpacyAnnotatorMetaData
-- that references the first SpacyToken (duui-spacy writes one per token). nil if there is none.
function spacyVectorsModel(inputCas)
    if not spacy_meta_ok then
        return nil
    end
    local first = nil
    local tokens_it = JCasUtil:select(inputCas, Token):iterator()
    while tokens_it:hasNext() do
        local token = tokens_it:next()
        if token:getType():getName() == SPACY_TOKEN_TYPE_NAME then
            first = token
            break
        end
    end
    if first == nil then
        return nil
    end
    local metas_it = JCasUtil:select(inputCas, SpacyAnnotatorMetaData):iterator()
    while metas_it:hasNext() do
        local meta = metas_it:next()
        local reference = meta:getReference()
        -- Lua proxies may not keep object identity, so compare with equals, then by type and span
        if reference ~= nil and (sameAnnotation(reference, first) or (
            reference:getType():getName() == first:getType():getName()
            and reference:getBegin() == first:getBegin()
            and reference:getEnd() == first:getEnd()
        )) then
            return meta:getModelName()
        end
    end
    return nil
end

-- The token vectors from the WordEmbedding annotations (README, "Vectors"): those whose
-- shortName is the vectors parameter, or, without the parameter, of the only shortName in
-- the CAS. Returns nil if the CAS has no WordEmbedding (then SpacyToken.vector is used),
-- else a table with the chosen name (nil if the parameter is missing and the CAS has
-- several or no names), all names in the CAS (comma-separated) and the embeddings by span
-- "begin:end", in CAS order. A token takes the embedding that references it; otherwise
-- the next unused one of its span, so that the two words of a split contraction (one span)
-- get their own vectors when the producer writes one embedding per token, in token order.
-- This is how StanzaTokens get their vectors: duui-spacy (token_type=StanzaToken) writes a
-- SpacyToken and its WordEmbedding per StanzaToken, in the order of the StanzaTokens.
function wordEmbeddings(inputCas, requested)
    if not word_embedding_ok then
        return nil
    end
    local embeddings_it = JCasUtil:select(inputCas, WordEmbedding):iterator()
    if not embeddings_it:hasNext() then
        return nil
    end
    local seen = {}
    local names = {}
    while embeddings_it:hasNext() do
        local name = embeddings_it:next():getShortName()
        if name ~= nil and not seen[name] then
            seen[name] = true
            names[#names + 1] = name
        end
    end
    local chosen = requested
    if chosen == nil or chosen == "" then
        chosen = nil
        if #names == 1 then
            chosen = names[1]
        end
    end
    local by_span = {}
    if chosen ~= nil then
        embeddings_it = JCasUtil:select(inputCas, WordEmbedding):iterator()
        while embeddings_it:hasNext() do
            local embedding = embeddings_it:next()
            if embedding:getShortName() == chosen then
                local span = embedding:getBegin() .. ":" .. embedding:getEnd()
                if by_span[span] == nil then
                    by_span[span] = {}
                end
                local list = by_span[span]
                list[#list + 1] = {reference = embedding:getReference(), embedding = embedding:getEmbedding(), used = false}
            end
        end
    end
    return {
        name = chosen,
        names = table.concat(names, ","),
        by_span = by_span,
    }
end

function tokenEmbedding(embeddings, token)
    local list = embeddings.by_span[token:getBegin() .. ":" .. token:getEnd()]
    if list == nil then
        return nil
    end
    for _, entry in ipairs(list) do
        if not entry.used and sameAnnotation(entry.reference, token) then
            entry.used = true
            return entry.embedding
        end
    end
    for _, entry in ipairs(list) do
        if not entry.used then
            entry.used = true
            return entry.embedding
        end
    end
    return nil
end

function serialize(inputCas, outputStream, parameters)
    local embeddings = wordEmbeddings(inputCas, parameters["vectors"])
    local token_layer = tokenLayer(inputCas)
    local sentence_layer = sentenceLayer(inputCas)
    local paragraphs = {}
    local paragraphs_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Paragraph)):listIterator()
    while paragraphs_it:hasNext() do
        local paragraph = paragraphs_it:next()
        local paragraph_data = {
            begin = paragraph:getBegin(),
            ['end'] = paragraph:getEnd(),
            text = paragraph:getCoveredText(),
            sentences = {}
        }
        local sentences_it = coveredOfType(Sentence, paragraph, sentence_layer):listIterator()
        while sentences_it:hasNext() do
            local sentence = sentences_it:next()
            local sentence_data = {
                begin = sentence:getBegin(),
                ['end'] = sentence:getEnd(),
                text = sentence:getCoveredText(),
                tokens = {}
            }
            local sentence_tokens = sentenceTokens(sentence, token_layer)
            local dependencies = luajava.newInstance("java.util.ArrayList", JCasUtil:selectCovered(Dependency, sentence))

            local tokens_it = sentence_tokens:listIterator()
            while tokens_it:hasNext() do
                -- the zero-based sentence-local token index, so that the governor can be
                -- transferred as head_index
                local token_index = tokens_it:nextIndex()
                local token = tokens_it:next()

                local dep_type = ""
                local head_index = nil
                local deps_it = dependencies:listIterator()
                while deps_it:hasNext() do
                    local dep = deps_it:next()
                    if sameAnnotation(dep:getDependent(), token) then
                        dep_type = dep:getDependencyType()
                        local governor = dep:getGovernor()
                        if governor ~= nil then
                            -- by identity: a span alone is ambiguous for the two words of a split contraction
                            for i = 0, sentence_tokens:size() - 1 do
                                if sameAnnotation(sentence_tokens:get(i), governor) then
                                    head_index = i
                                    break
                                end
                            end
                            -- a span match only when it identifies exactly one sentence token
                            if head_index == nil then
                                local matching_index = nil
                                local matching_count = 0
                                for i = 0, sentence_tokens:size() - 1 do
                                    local candidate = sentence_tokens:get(i)
                                    if candidate:getBegin() == governor:getBegin()
                                        and candidate:getEnd() == governor:getEnd()
                                    then
                                        matching_index = i
                                        matching_count = matching_count + 1
                                    end
                                end
                                if matching_count == 1 then
                                    head_index = matching_index
                                end
                            end
                        end
                        -- the root, if no governor could be resolved
                        if head_index == nil and (dep_type == "--" or dep_type == "ROOT" or dep_type == "root") then
                            head_index = token_index
                        end
                        break
                    end
                end

                local vector = nil
                local vector_array = nil
                if embeddings ~= nil then
                    vector_array = tokenEmbedding(embeddings, token)
                elseif token:getType():getName() == SPACY_TOKEN_TYPE_NAME and token:getHasVector() then
                    -- nil if duui-spacy wrote no vectors (write_vectors=false)
                    vector_array = token:getVector()
                end
                local has_vector = vector_array ~= nil
                if has_vector then
                    vector = {}
                    local vector_it = vector_array:iterator()
                    while vector_it:hasNext() do
                        vector[#vector + 1] = vector_it:next()
                    end
                end

                local pos = token:getPos()
                local lemma = token:getLemma()
                local morph = token:getMorph()
                local token_data = {
                    begin = token:getBegin(),
                    ['end'] = token:getEnd(),
                    -- is_alpha / is_punct are derived from the word in duui.py
                    text = tokenWord(token),
                    lemma = lemma ~= nil and lemma:getValue() or nil,
                    pos_value = pos ~= nil and pos:getPosValue() or nil,
                    pos_coarse = pos ~= nil and pos:getCoarseValue() or nil,
                    dep_type = dep_type,
                    head_index = head_index,
                    morph_person = morph ~= nil and morphFeature(morph, morph:getPerson(), "Person") or nil,
                    morph_number = morph ~= nil and morphFeature(morph, morph:getNumber(), "Number") or nil,
                    morph_tense = morph ~= nil and morphFeature(morph, morph:getTense(), "Tense") or nil,
                    morph_verbform = morph ~= nil and morphFeature(morph, morph:getVerbForm(), "VerbForm") or nil,
                    morph_mood = morph ~= nil and morphFeature(morph, morph:getMood(), "Mood") or nil,
                    morph_number_psor = morphFeature(morph, nil, "Number[psor]"),
                    vector = vector,
                    has_vector = has_vector,
                }
                sentence_data.tokens[#sentence_data.tokens + 1] = token_data
            end
            paragraph_data.sentences[#paragraph_data.sentences + 1] = sentence_data
        end
        paragraphs[#paragraphs + 1] = paragraph_data
    end

    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        language = inputCas:getDocumentLanguage(),
        paragraphs = paragraphs,
        -- optional, comma-separated index names (README, "Selecting indices")
        include_indices = parameters["include_indices"],
        exclude_indices = parameters["exclude_indices"],
        -- source of the token vectors (README, "Vectors"): the shortName of the
        -- WordEmbeddings, or without WordEmbeddings SpacyToken.vector with the
        -- optional parameter (default SPACY) and the spaCy model
        vectors_source = embeddings ~= nil and "WordEmbedding" or "SpacyToken",
        vectors = embeddings ~= nil and embeddings.name or parameters["vectors"],
        vectors_available = embeddings ~= nil and embeddings.names or nil,
        vectors_model = embeddings == nil and spacyVectorsModel(inputCas) or nil,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["indices"] ~= nil then
        local meta = results["meta"]

        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local cas = inputCas:getCas()
        local type_system = cas:getTypeSystem()

        for i, index in ipairs(results["indices"]) do
            local index_type_name = resolveIndexTypeName(index)
            local index_type = type_system:getType(index_type_name)

            if index_type == nil then
                error(
                    "Coh-Metrix UIMA type is missing from TypeSystem.xml: "
                    .. index_type_name
                )
            end

            -- Create the subtype dynamically through the CAS API. This avoids
            -- requiring one generated Java/JCas class for every Coh-Metrix
            -- index while retaining all features inherited from Index.
            local index_anno = cas:createAnnotation(index_type, 0, doc_len)

            local index_feature = requireFeature(index_type, "index")
            local type_name_feature = requireFeature(index_type, "typeName")
            local label_ttlab_feature = requireFeature(index_type, "labelTTLab")
            local label_v3_feature = requireFeature(index_type, "labelV3")
            local label_v2_feature = requireFeature(index_type, "labelV2")
            local description_feature = requireFeature(index_type, "description")
            local value_feature = requireFeature(index_type, "value")
            local error_feature = requireFeature(index_type, "error")
            local version_feature = requireFeature(index_type, "version")

            index_anno:setIntValue(index_feature, index["index"])
            setOptionalStringFeature(index_anno, type_name_feature, index["type_name"])
            setOptionalStringFeature(index_anno, label_ttlab_feature, index["label_ttlab"])
            setOptionalStringFeature(index_anno, label_v3_feature, index["label_v3"])
            setOptionalStringFeature(index_anno, label_v2_feature, index["label_v2"])
            setOptionalStringFeature(index_anno, description_feature, index["description"])
            -- UIMA's primitive double feature cannot represent JSON null.
            -- Preserve "not computable" as NaN so it cannot silently become
            -- the valid, calculated result 0.0 in the CAS.
            if index["value"] == nil then
                index_anno:setDoubleValue(value_feature, Double.NaN)
            else
                index_anno:setDoubleValue(value_feature, index["value"])
            end
            setOptionalStringFeature(index_anno, error_feature, index["error"])
            setOptionalStringFeature(index_anno, version_feature, index["version"])
            cas:addFsToIndexes(index_anno)

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(index_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
-- Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Token = luajava.bindClass("org.texttechnologylab.uima.type.spacy.SpacyToken")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: Table/Dictonary of parameters that should be used to configure the annotator
function serialize(inputCas, outputStream, parameters)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local doc_text = inputCas:getDocumentText()
    local doc_lang = inputCas:getDocumentLanguage()

    -- Should use tokens directly?
    local tokens = nil
    local spaces = nil
    local sent_starts = nil
    local use_existing_tokens = false
    local use_existing_sentences = false
    if parameters["use_existing_tokens"] ~= nil then
        use_existing_tokens = parameters["use_existing_tokens"] == "true"
    end
    if parameters["use_existing_sentences"] ~= nil then
        use_existing_sentences = parameters["use_existing_sentences"] == "true"
    end
    if use_existing_tokens then
        tokens = {}
        spaces = {}
        sent_starts = {}

        local tokens_count = 1
        local tokens_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Token)):listIterator()
        local sentences = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Sentence))
        while tokens_it:hasNext() do
            local token = tokens_it:next()
            tokens[tokens_count] = token:getCoveredText()
            -- try to get next to see if space is needed
            has_space = false
            if tokens_it:hasNext() then
                local next_token = tokens_it:next()
                has_space = next_token:getBegin() ~= token:getEnd()
                tokens_it:previous()
            end
            spaces[tokens_count] = has_space
            if use_existing_sentences then
                local sentences_it = sentences:listIterator()
                sent_starts[tokens_count] = false
                while sentences_it:hasNext() do
                    local sentence = sentences_it:next()
                    if sentence:getBegin() == token:getBegin() then
                        sent_starts[tokens_count] = true
                        break
                    elseif sentence:getBegin() > token:getBegin() then
                        break
                    end
                end
            end

            tokens_count = tokens_count + 1
        end

        -- reset text
        doc_text = ""
    end

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        parameters = parameters,
        tokens = tokens,
        spaces = spaces,
        sent_starts = sent_starts
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

    -- Add modification annotation
    local modification_meta = results["modification_meta"]
    local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
    modification_anno:setUser(modification_meta["user"])
    modification_anno:setTimestamp(modification_meta["timestamp"])
    modification_anno:setComment(modification_meta["comment"])
    modification_anno:addToIndexes()

    -- Get meta data, this is the same for every annotation
    local meta = results["meta"]

    -- If was pretokenized, use existing tokens
    local is_pretokenized = results["is_pretokenized"]

    -- Add sentences
    for i, sent in ipairs(results["sentences"]) do
        -- Writing can be disabled via parameters
        -- Note: spaCy will still run the full pipeline, and all results are based on these results
        if sent["write_sentence"] then
            -- Create sentence annotation
            local sent_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", inputCas)
            sent_anno:setBegin(sent["begin"])
            sent_anno:setEnd(sent["end"])
            sent_anno:addToIndexes()

            -- Create annotator meta data annotation, using the base meta data
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(sent_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end
    end

    -- Add tokens
    -- Save all tokens, to allow for retrieval in dependencies
    local all_tokens = {}
    if is_pretokenized then
        local tokens_count = 0
        local tokens_it = JCasUtil:select(inputCas, Token):iterator()
        while tokens_it:hasNext() do
            local token = tokens_it:next()
            all_tokens[tokens_count] = token
            tokens_count = tokens_count + 1
        end
    end
    for i, token in ipairs(results["tokens"]) do
        -- Save current token
        local token_anno = nil
        if is_pretokenized then
            -- Use existing token if pretokenized
            token_anno = all_tokens[i-1]
        elseif token["write_token"] then
            -- Create token annotation
--             token_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", inputCas)
            token_anno = luajava.newInstance("org.texttechnologylab.uima.type.spacy.SpacyToken", inputCas)
            token_anno:setBegin(token["begin"])
            token_anno:setEnd(token["end"])

            -- spacy token extras
            token_anno:setLikeUrl(token["like_url"])
            token_anno:setHasVector(token["has_vector"])
            token_anno:setLikeNum(token["like_num"])
            token_anno:setIsStop(token["is_stop"])
            token_anno:setIsOov(token["is_oov"])
            token_anno:setIsCurrency(token["is_currency"])
            token_anno:setIsQuote(token["is_quote"])
            token_anno:setIsBracket(token["is_bracket"])
            token_anno:setIsSentStart(token["is_sent_start"])
            token_anno:setIsSentEnd(token["is_sent_end"])
            token_anno:setIsLeftPunct(token["is_left_punct"])
            token_anno:setIsRightPunct(token["is_right_punct"])
            token_anno:setIsPunct(token["is_punct"])
            token_anno:setIsTitle(token["is_title"])
            token_anno:setIsUpper(token["is_upper"])
            token_anno:setIsLower(token["is_lower"])
            token_anno:setIsDigit(token["is_digit"])
            token_anno:setIsAscii(token["is_ascii"])
            token_anno:setIsAlpha(token["is_alpha"])

            if token["vector"] ~= nil then
                local vector_length = #token["vector"]
                token_anno:setVector(luajava.newInstance("org.apache.uima.jcas.cas.FloatArray", inputCas, vector_length))
                for vector_ind, vector_val in ipairs(token["vector"]) do
                    -- Note: Lua starts counting at 1, but Java at 0
                    token_anno:setVector(vector_ind-1, vector_val)
                end
            end

            -- benepar
            if token["benepar_labels"] ~= nil then
                local benepar_labels_length = #token["benepar_labels"]
                if benepar_labels_length > 0 then
                    token_anno:setBeneparLabels(luajava.newInstance("org.apache.uima.jcas.cas.StringArray", inputCas, benepar_labels_length))
                    for label_ind, label_val in ipairs(token["benepar_labels"]) do
                        token_anno:setBeneparLabels(label_ind-1, label_val)
                    end
                end
            end

            token_anno:addToIndexes()

            -- URL detection
            if token["like_url"] then
                url_anno = luajava.newInstance("org.texttechnologylab.type.id.URL", inputCas)
                url_anno:setBegin(token["begin"])
                url_anno:setEnd(token["end"])

                -- optional url might be split in parts
                if token["url_parts"] ~= nil then
                    url_anno:setScheme(token["url_parts"]["scheme"])
                    url_anno:setUser(token["url_parts"]["user"])
                    url_anno:setPassword(token["url_parts"]["password"])
                    url_anno:setHost(token["url_parts"]["host"])
                    url_anno:setPort(token["url_parts"]["port"])
                    url_anno:setPath(token["url_parts"]["path"])
                    url_anno:setQuery(token["url_parts"]["query"])
                    url_anno:setFragment(token["url_parts"]["fragment"])
                end
                url_anno:addToIndexes()
            end

            -- Save current token using its index
            -- Note: Lua starts counting at 1
            all_tokens[i-1] = token_anno

            -- Create meta data for this token
            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(token_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end

        if token["write_lemma"] then
            local lemma_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma", inputCas)
            lemma_anno:setBegin(token["begin"])
            lemma_anno:setEnd(token["end"])
            if token["lemma"] == nil or token["lemma"] == "" then
                if token_anno ~= nil then
                    lemma_anno:setValue(token_anno:getCoveredText())
                end
            else
                lemma_anno:setValue(token["lemma"])
            end
            lemma_anno:addToIndexes()

            -- If there is a token, i.e. writing is not disabled for tokens, add this lemma infos to the token
            if token_anno ~= nil then
                token_anno:setLemma(lemma_anno)
            end

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(lemma_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end

        if token["write_pos"] then
            -- TODO Add full pos mapping for different pos types
            local pos_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", inputCas)
            pos_anno:setBegin(token["begin"])
            pos_anno:setEnd(token["end"])
            pos_anno:setPosValue(token["pos"])
            pos_anno:setCoarseValue(token["pos_coarse"])
            pos_anno:addToIndexes()

            if token_anno ~= nil then
                token_anno:setPos(pos_anno)
            end

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(pos_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end

        if token["write_morph"] then
            local morph_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", inputCas)
            morph_anno:setBegin(token["begin"])
            morph_anno:setEnd(token["end"])
            morph_anno:setValue(token["morph"])

            -- Add detailed infos, if available
            if token["morph_details"]["gender"] ~= nil then
                morph_anno:setGender(token["morph_details"]["gender"])
            end
            if token["morph_details"]["number"] ~= nil then
                morph_anno:setNumber(token["morph_details"]["number"])
            end
            if token["morph_details"]["case"] ~= nil then
                morph_anno:setCase(token["morph_details"]["case"])
            end
            if token["morph_details"]["degree"] ~= nil then
                morph_anno:setDegree(token["morph_details"]["degree"])
            end
            if token["morph_details"]["verbForm"] ~= nil then
                morph_anno:setVerbForm(token["morph_details"]["verbForm"])
            end
            if token["morph_details"]["tense"] ~= nil then
                morph_anno:setTense(token["morph_details"]["tense"])
            end
            if token["morph_details"]["mood"] ~= nil then
                morph_anno:setMood(token["morph_details"]["mood"])
            end
            if token["morph_details"]["voice"] ~= nil then
                morph_anno:setVoice(token["morph_details"]["voice"])
            end
            if token["morph_details"]["definiteness"] ~= nil then
                morph_anno:setDefiniteness(token["morph_details"]["definiteness"])
            end
            if token["morph_details"]["person"] ~= nil then
                morph_anno:setPerson(token["morph_details"]["person"])
            end
            if token["morph_details"]["aspect"] ~= nil then
                morph_anno:setAspect(token["morph_details"]["aspect"])
            end
            if token["morph_details"]["animacy"] ~= nil then
                morph_anno:setAnimacy(token["morph_details"]["animacy"])
            end
            if token["morph_details"]["gender"] ~= nil then
                morph_anno:setNegative(token["morph_details"]["negative"])
            end
            if token["morph_details"]["numType"] ~= nil then
                morph_anno:setNumType(token["morph_details"]["numType"])
            end
            if token["morph_details"]["possessive"] ~= nil then
                morph_anno:setPossessive(token["morph_details"]["possessive"])
            end
            if token["morph_details"]["pronType"] ~= nil then
                morph_anno:setPronType(token["morph_details"]["pronType"])
            end
            if token["morph_details"]["reflex"] ~= nil then
                morph_anno:setReflex(token["morph_details"]["reflex"])
            end
            if token["morph_details"]["transitivity"] ~= nil then
                morph_anno:setTransitivity(token["morph_details"]["transitivity"])
            end

            morph_anno:addToIndexes()

            if token_anno ~= nil then
                token_anno:setMorph(morph_anno)
            end

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(morph_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end
    end

    -- Add dependencies
    for i, dep in ipairs(results["dependencies"]) do
        if dep["write_dep"] then
            -- Create specific annotation based on type
            local dep_anno
            if dep["type"] == "ROOT" then
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT", inputCas)
                dep_anno:setDependencyType("--")
            else
                dep_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency", inputCas)
                dep_anno:setDependencyType(dep["type"])
            end

            dep_anno:setBegin(dep["begin"])
            dep_anno:setEnd(dep["end"])
            dep_anno:setFlavor(dep["flavor"])

            -- Get needed tokens via indices
            governor_token = all_tokens[dep["governor_ind"]]
            if governor_token ~= nil then
                dep_anno:setGovernor(governor_token)
            end

            dependent_token = all_tokens[dep["dependent_ind"]]
            if governor_token ~= nil then
                dep_anno:setDependent(dependent_token)
            end

            if governor_token ~= nil and dependent_token ~= nil then
                dependent_token:setParent(governor_token)
            end

            dep_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(dep_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end
    end

    -- Add entities
    for i, ent in ipairs(results["entities"]) do
        if ent["write_entity"] then
            local ent_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", inputCas)
            ent_anno:setBegin(ent["begin"])
            ent_anno:setEnd(ent["end"])
            ent_anno:setValue(ent["value"])
            ent_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
            meta_anno:setReference(ent_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:setSpacyVersion(meta["spacyVersion"])
            meta_anno:setModelLang(meta["modelLang"])
            meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
            meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
            meta_anno:addToIndexes()
        end
    end

    -- Add noun chunks
    for i, nc in ipairs(results["noun_chunks"]) do
        local nc_anno = luajava.newInstance("org.texttechnologylab.uima.type.spacy.SpacyNounChunk", inputCas)
        nc_anno:setBegin(nc["begin"])
        nc_anno:setEnd(nc["end"])
        nc_anno:addToIndexes()

        local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", inputCas)
        meta_anno:setReference(ent_anno)
        meta_anno:setName(meta["name"])
        meta_anno:setVersion(meta["version"])
        meta_anno:setModelName(meta["modelName"])
        meta_anno:setModelVersion(meta["modelVersion"])
        meta_anno:setSpacyVersion(meta["spacyVersion"])
        meta_anno:setModelLang(meta["modelLang"])
        meta_anno:setModelSpacyVersion(meta["modelSpacyVersion"])
        meta_anno:setModelSpacyGitVersion(meta["modelSpacyGitVersion"])
        meta_anno:addToIndexes()
    end
end

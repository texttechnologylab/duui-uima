-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")

------------------------------------------------------

---Indicates that this component supports the "new" `process` method.
SUPPORTS_PROCESS = true
---Indicates that this component does NOT support the old `serialize`/`deserialize` methods.
SUPPORTS_SERIALIZE = false

------------------------------------------------------
--- Below are two general purpose functions for batch processing of annotations in a JCas.
--- These are not specific for this component, but only support a simple Iterator as input.

---Create and yield batches of elements from an iterator after applying a transform function.
---@param iterator any an iterator over annotations
---@param transform fun(any): any a tranform function over the elements of the iterator
---@param batch_size integer size of each batch sent to the component
function get_batches(iterator, transform, batch_size)
    local batch = {}
    while iterator:hasNext() do
        batch[#batch + 1] = transform(iterator:next())
        if #batch == batch_size then
            coroutine.yield(batch)
            batch = {}
        end
    end

    if #batch > 0 then
        coroutine.yield(batch)
    end
end

---Iterate over batches of elements from an iterator after applying a transform function.
---@param iterator any an iterator over annotations
---@param transform fun(any): any a tranform function over the elements of the iterator
---@param batch_size integer size of each batch
---@return fun(): table an iterator over batches to process
function batched(iterator, transform, batch_size)
    local co = coroutine.create(function() get_batches(iterator, transform, batch_size) end)
    return function()
        local _, batch = coroutine.resume(co)
        return batch
    end
end

------------------------------------------------------

---Get the text and offset (begin index) of a sentence.
---@param sentence any a sentence annotation
---@return table a table with the text and offset of the sentence
function get_sentence_and_offset(sentence)
    return {
        text = sentence:getCoveredText(),
        offset = sentence:getBegin(),
    }
end

---We can define settings here or we could fetch them from the component using separate endpoints.
REQUEST_BATCH_SIZE = 1024

---Process the sentences in the given JCas in small batches.
---@param sourceCas any JCas (view) to process
---@param handler any DuuiHttpRequestHandler with a connection to the running component
---@param parameters table<string, string> optional parameters
---@param targetCas any JCas (view) to write the results to (optional)
function process(sourceCas, handler, parameters, targetCas)
    parameters = parameters or {}
    local config = {
        spacy_language = sourceCas:getDocumentLanguage(),
        spacy_model_size = parameters.spacy_model_size or "lg",
        spacy_batch_size = tonumber(parameters.spacy_batch_size) or 32,
    }

    targetCas = targetCas or sourceCas

    ---If there are no sentences in the source JCas (view), we can call the supplementary /eos
    ---endpoint of the component to annotate them. The spaCy `senter` pipeline component can deal
    ---with much larger inputs than the other components, so we can use the whole document text.
    local sentences = JCasUtil:select(sourceCas, Sentence)

    if sentences:isEmpty() then
        local response = handler:post("/eos", json.encode({
            text = sourceCas:getDocumentText(),
            config = config,
        }))
        process_eos(targetCas, response)
        sentences = JCasUtil:select(targetCas, Sentence)

        if sentences:isEmpty() then
            error("No sentences found in the source or target JCas.")
        end
    end

    ---After fetching the sentences (and possibly annotating them), we can process them in batches.
    ---The batch size is variable, here we use a fixed batch size in number of sentences.
    ---Developers could also implement dynamic batch sizes depending on information provided by the
    ---component or on the length of the text etc.
    ---Using the `get_sentence_and_offset` transform function, we just get the text and offset of each
    ---sentence. The general purpose batching functions from above deal with the rest.
    ---After the component has processed the sentences, we call `process_response` on the response directly.

    local batch_size = parameters.request_batch_size or REQUEST_BATCH_SIZE
    ---@type table<integer, any> table to aggregate references to created annotations
    local references = {}
    ---@type table<string, table>
    local results = {}
    for batch in batched(sentences:iterator(), get_sentence_and_offset, batch_size) do
        if type(batch) ~= "table" then
            error("Error while batching: " .. batch)
        end

        ---@type any DuuiHttpRequestHandler.Response{int statusCode, byte[]? body}
        local response = handler:process(
            json.encode({
                sentences = batch,
                config = config,
            })
        )

        ---The response wraps the HTTP status code and body in a record class.
        ---If there is an error, we can deal with it here. We could also make use of additional information
        ---provided by the component, e.g. the error message in the body.
        ---Here, we just throw an error with the status code and body.
        
        if response:statusCode() ~= 200 then
            error("Error " .. response:statusCode() .. " in communication with component: " .. response:bodyUtf8())
        end

        ---The Response object provides a method to decode the body as UTF-8, which we then decode as JSON.
        results = json.decode(response:bodyUtf8())

        ---We collect all annotation references in a single table to deduplicate the annotator metadata annotation.

        local batch_refs = process_response(targetCas, results)
        for _, ref in ipairs(batch_refs) do
            references[#references + 1] = ref
        end
    end

    ---After processing all sentences, we can add the metadata to the target JCas.
    ---The metadata is provided by the component in the response, so we can just use it here.

    add_annotator_metadata(targetCas, results.metadata, references)
end

---Process the response from the component.
---@param targetCas any JCas to write the results to
---@param results table the results from the component
function process_response(targetCas, results)
    ---Below follows basic DUUI deserialization logic, adapted from the regular duui-spacy component.

    local tokens, references = {}, {}

    for _, token in ipairs(results.tokens) do
        local token_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", targetCas)
        token_anno:setBegin(token["begin"])
        token_anno:setEnd(token["end"])
        token_anno:addToIndexes()

        tokens[#tokens + 1] = token_anno
        references[#references + 1] = token_anno

        if token.lemma ~= nil then
            local lemma_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma", targetCas)
            lemma_anno:setBegin(token["begin"])
            lemma_anno:setEnd(token["end"])
            lemma_anno:setValue(token.lemma)
            token_anno:setLemma(lemma_anno)
            lemma_anno:addToIndexes()

            references[#references + 1] = lemma_anno
        end

        if token.pos_value ~= nil then
            local pos_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS", targetCas)
            pos_anno:setBegin(token["begin"])
            pos_anno:setEnd(token["end"])
            pos_anno:setPosValue(token.pos_value)
            pos_anno:setCoarseValue(token.pos_coarse)
            token_anno:setPos(pos_anno)
            pos_anno:addToIndexes()

            references[#references + 1] = pos_anno
        end

        if token.morph_value ~= nil and token.morph_value ~= "" then
            local morph_anno = luajava.newInstance(
                "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.morph.MorphologicalFeatures", targetCas
            )
            morph_anno:setBegin(token["begin"])
            morph_anno:setEnd(token["end"])
            morph_anno:setValue(token.morph_value)
            token_anno:setMorph(morph_anno)
            morph_anno:addToIndexes()

            references[#references + 1] = morph_anno

            -- Add detailed infos, if available
            if token.morph_features.Gender ~= nil then
                morph_anno:setGender(token.morph_features.Gender)
            end
            if token.morph_features.Number ~= nil then
                morph_anno:setNumber(token.morph_features.Number)
            end
            if token.morph_features.Case ~= nil then
                morph_anno:setCase(token.morph_features.Case)
            end
            if token.morph_features.Degree ~= nil then
                morph_anno:setDegree(token.morph_features.Degree)
            end
            if token.morph_features.VerbForm ~= nil then
                morph_anno:setVerbForm(token.morph_features.VerbForm)
            end
            if token.morph_features.Tense ~= nil then
                morph_anno:setTense(token.morph_features.Tense)
            end
            if token.morph_features.Mood ~= nil then
                morph_anno:setMood(token.morph_features.Mood)
            end
            if token.morph_features.Voice ~= nil then
                morph_anno:setVoice(token.morph_features.Voice)
            end
            if token.morph_features.Definiteness ~= nil then
                morph_anno:setDefiniteness(token.morph_features.Definiteness)
            end
            if token.morph_features.Person ~= nil then
                morph_anno:setPerson(token.morph_features.Person)
            end
            if token.morph_features.Aspect ~= nil then
                morph_anno:setAspect(token.morph_features.Aspect)
            end
            if token.morph_features.Animacy ~= nil then
                morph_anno:setAnimacy(token.morph_features.Animacy)
            end
            if token.morph_features.Negative ~= nil then
                morph_anno:setNegative(token.morph_features.Negative)
            end
            if token.morph_features.NumType ~= nil then
                morph_anno:setNumType(token.morph_features.NumType)
            end
            if token.morph_features.Possessive ~= nil then
                morph_anno:setPossessive(token.morph_features.Possessive)
            end
            if token.morph_features.PronType ~= nil then
                morph_anno:setPronType(token.morph_features.PronType)
            end
            if token.morph_features.Reflex ~= nil then
                morph_anno:setReflex(token.morph_features.Reflex)
            end
            if token.morph_features.Transitivity ~= nil then
                morph_anno:setTransitivity(token.morph_features.Transitivity)
            end
        end
    end

    for _, dep in ipairs(results.dependencies) do
        local dep_type = dep.dependency_type
        local DEP_TYPE = string.upper(dep_type)
        local dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency"
        if DEP_TYPE == "ROOT" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT"
            dep_type = "--"
        elseif DEP_TYPE == "ABBREV" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ABBREV"
        elseif DEP_TYPE == "ACOMP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ACOMP"
        elseif DEP_TYPE == "ADVCL" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVCL"
        elseif DEP_TYPE == "ADVMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ADVMOD"
        elseif DEP_TYPE == "AGENT" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AGENT"
        elseif DEP_TYPE == "AMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AMOD"
        elseif DEP_TYPE == "APPOS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.APPOS"
        elseif DEP_TYPE == "ATTR" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ATTR"
        elseif DEP_TYPE == "AUX0" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUX0"
        elseif DEP_TYPE == "AUXPASS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.AUXPASS"
        elseif DEP_TYPE == "CC" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CC"
        elseif DEP_TYPE == "CCOMP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CCOMP"
        elseif DEP_TYPE == "COMPLM" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COMPLM"
        elseif DEP_TYPE == "CONJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ"
        elseif DEP_TYPE == "CONJP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJP"
        elseif DEP_TYPE == "CONJ_YET" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CONJ_YET"
        elseif DEP_TYPE == "COP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.COP"
        elseif DEP_TYPE == "CSUBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJ"
        elseif DEP_TYPE == "CSUBJPASS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.CSUBJPASS"
        elseif DEP_TYPE == "DEP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DEP"
        elseif DEP_TYPE == "DET" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DET"
        elseif DEP_TYPE == "DOBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.DOBJ"
        elseif DEP_TYPE == "EXPL" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.EXPL"
        elseif DEP_TYPE == "INFMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.INFMOD"
        elseif DEP_TYPE == "IOBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.IOBJ"
        elseif DEP_TYPE == "MARK" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MARK"
        elseif DEP_TYPE == "MEASURE" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MEASURE"
        elseif DEP_TYPE == "MWE" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.MWE"
        elseif DEP_TYPE == "NEG" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NEG"
        elseif DEP_TYPE == "NN" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NN"
        elseif DEP_TYPE == "NPADVMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NPADVMOD"
        elseif DEP_TYPE == "NSUBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJ"
        elseif DEP_TYPE == "NSUBJPASS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NSUBJPASS"
        elseif DEP_TYPE == "NUM" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUM"
        elseif DEP_TYPE == "NUMBER" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.NUMBER"
        elseif DEP_TYPE == "PARATAXIS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARATAXIS"
        elseif DEP_TYPE == "PARTMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PARTMOD"
        elseif DEP_TYPE == "PCOMP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PCOMP"
        elseif DEP_TYPE == "POBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POBJ"
        elseif DEP_TYPE == "POSS" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSS"
        elseif DEP_TYPE == "POSSESSIVE" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.POSSESSIVE"
        elseif DEP_TYPE == "PRECONJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRECONJ"
        elseif DEP_TYPE == "PRED" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRED"
        elseif DEP_TYPE == "PREDET" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREDET"
        elseif DEP_TYPE == "PREP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREP"
        elseif DEP_TYPE == "PREPC" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PREPC"
        elseif DEP_TYPE == "PRT" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PRT"
        elseif DEP_TYPE == "PUNCT" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PUNCT"
        elseif DEP_TYPE == "PURPCL" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.PURPCL"
        elseif DEP_TYPE == "QUANTMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.QUANTMOD"
        elseif DEP_TYPE == "RCMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.RCMOD"
        elseif DEP_TYPE == "REF" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REF"
        elseif DEP_TYPE == "REL" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.REL"
        elseif DEP_TYPE == "ROOT" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT"
        elseif DEP_TYPE == "TMOD" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.TMOD"
        elseif DEP_TYPE == "XSUBJ" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XSUBJ"
        elseif DEP_TYPE == "XCOMP" then
            dep_anno_type = "de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.XCOMP"
        end
        local dep_anno = luajava.newInstance(dep_anno_type, targetCas)
        dep_anno:setDependencyType(dep_type)

        dep_anno:setBegin(dep["begin"])
        dep_anno:setEnd(dep["end"])
        dep_anno:setFlavor(dep.flavor)

        local governor = tokens[dep.governor_index + 1]
        if governor ~= nil then
            dep_anno:setGovernor(governor)
        end

        local dependent = tokens[dep.dependent_index + 1]
        if dependent ~= nil then
            dep_anno:setDependent(dependent)
        end

        if governor ~= nil and dependent ~= nil then
            dependent:setParent(governor)
        end
        dep_anno:addToIndexes()

        references[#references + 1] = dep_anno
    end

    for _, entity in ipairs(results.entities) do
        local entity_anno

        local entity_value = entity.value
        local ENTITY_VALUE = string.upper(entity_value)
        if entity_value == "Organization" or ENTITY_VALUE == "ORG" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Organization", targetCas)
        elseif entity_value == "Person" or ENTITY_VALUE == "PER" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Person", targetCas)
        elseif entity_value == "Location" or ENTITY_VALUE == "LOC" then
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location", targetCas)
        else
            entity_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity", targetCas)
        end

        entity_anno:setBegin(entity["begin"])
        entity_anno:setEnd(entity["end"])
        entity_anno:setValue(entity_value)
        entity_anno:addToIndexes()

        references[#references + 1] = entity_anno
    end

    return references
end

function process_eos(targetCas, response)
    if response:statusCode() ~= 200 then
        error("Error " .. response:statusCode() .. " in communication with component: " .. response:body())
    end

    local results = json.decode(response:bodyUtf8())

    local references = {}
    for _, sentence in ipairs(results.sentences) do
        local sentence_anno = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence",
            targetCas)
        sentence_anno:setBegin(sentence["begin"])
        sentence_anno:setEnd(sentence["end"])
        sentence_anno:addToIndexes()

        references[#references + 1] = sentence_anno
    end

    add_annotator_metadata(targetCas, results.metadata, references)
end

---Add a SpacyAnnotatorMetaData annotation to the targetCas.
---@param targetCas any the JCas to add the annotation to
---@param metadata table<string, any> a table with metadata information
---@param references table<integer, any> a table of references to the annotations
function add_annotator_metadata(targetCas, metadata, references)
    local reference_array = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", targetCas, #references)
    for i, ref in ipairs(references) do
        reference_array:set(i - 1, ref)
    end

    local annotation = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", targetCas)
    annotation:setReference(reference_array)
    annotation:setName(metadata.name)
    annotation:setVersion(metadata.version)
    annotation:setSpacyVersion(metadata.spacy_version)
    annotation:setModelName(metadata.model_name)
    annotation:setModelVersion(metadata.model_version)
    annotation:setModelLang(metadata.model_lang)
    annotation:setModelSpacyVersion(metadata.model_spacy_version)
    annotation:setModelSpacyGitVersion(metadata.model_spacy_git_version)
    annotation:addToIndexes()
end

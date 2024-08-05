-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Entity = luajava.bindClass("org.texttechnologylab.annotation.semaf.isobase.Entity")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    local doc_text = inputCas:getDocumentText(str)
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
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
--     Get Token results
    local entities = results["entities"]
    cas_entities = {}

    for i, ent in ipairs(entities) do
        if ent["write"] then
            b = ent["begin"]
            e = ent["end"]

            local cas_ent = nil

            local eCheck = util:selectAt(inputCas, Entity, b, e):count()

            if eCheck == 1 then
                cas_ent = util:selectSingleAt(inputCas, Entity, b, e)
            else
                cas_ent = luajava.newInstance("org.texttechnologylab.annotation.semaf.isobase.Entity", inputCas)
                cas_ent:setBegin(b)
                cas_ent:setEnd(e)
                cas_ent:addToIndexes()
            end

            cas_entities[b] = {}
            cas_entities[b][e] = cas_ent

--             local cas_ent_p = luajava.newInstance("org.texttechnologylab.annotation.AnnotationPerspective", inputCas)
--             cas_ent_p:setName("srl")
--             cas_ent_p:setReference(cas_ent)
--             cas_ent_p:addToIndexes()

        end
    end

    local links = results["links"]
    for i, link in ipairs(links) do
        if link["write"] then
            local cas_link = luajava.newInstance("org.texttechnologylab.annotation.semaf.semafsr.SrLink", inputCas)
            b_f = link["begin_fig"]
            e_f = link["end_fig"]
            b_g = link["begin_gr"]
            e_g = link["end_gr"]
            cas_link:setFigure(cas_entities[b_f][e_f])
            cas_link:setGround(cas_entities[b_g][e_g])
            cas_link:setRel_type(link["rel_type"])
            cas_link:addToIndexes()

--             local cas_link_p = luajava.newInstance("org.texttechnologylab.annotation.AnnotationPerspective", inputCas)
--             cas_link_p:setName("bfsrl")
--             cas_link_p:setReference(cas_link)
--             cas_link_p:addToIndexes()
        end
    end
end

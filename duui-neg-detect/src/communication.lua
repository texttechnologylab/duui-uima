-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DocumentMetaData = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Lemma = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Negation = luajava.bindClass("org.texttechnologylab.annotation.negation.CompleteNegation")
Pos = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS")
FSArray = luajava.bindClass("org.apache.uima.jcas.cas.FSArray")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS
    -- For spaCy, we need the documents text and its language
    -- TODO add additional params?
    -- local doc_lang = inputCas:getDocumentLanguage()
    -- local doc_text = inputCas:getDocumentText()
    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts

    local cas_sentences = {}
    local tokens = {}
    local sent_counter = 1
    local tokens_counter = 1
    local sents = JCasUtil:select(inputCas, Sentence):iterator()
    while sents:hasNext() do
        local sent = sents:next()
        cas_sentences[sent_counter] = {}
        cas_sentences[sent_counter]["begin"] = sent:getBegin()
        cas_sentences[sent_counter]["end"] = sent:getEnd()
        cas_sentences[sent_counter]["coveredText"] = sent:getCoveredText()
        sent_counter = sent_counter + 1

        local token = {}
        local toks = JCasUtil:selectCovered(Token, sent):iterator()
        local token_counter = 1
        while toks:hasNext() do
            local tok = toks:next()
            token[token_counter] = {}
            token[token_counter]["begin"] = tok:getBegin()
            token[token_counter]["end"] = tok:getEnd()
            token[token_counter]["coveredText"] = tok:getCoveredText()
            token_counter = token_counter + 1
        end
        tokens[tokens_counter] = token
        tokens_counter = tokens_counter + 1

    end
    outputStream:write(json.encode({
        sentences = cas_sentences,
        tokens = tokens
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    --
    local results = json.decode(inputString)

    local token_list = {}
    local toks = JCasUtil:select(inputCas, Token):iterator()
    local token_counter = 1
    while toks:hasNext() do
        local tok = toks:next()
        token_list[token_counter] = tok
        token_counter = token_counter + 1
    end

    -- Negations
    local neg_list = {}
    local neg_counter = 1
    if results["negations"] ~= null then
        for i, neg in ipairs(results["negations"]) do
            local neg_obj = luajava.newInstance("org.texttechnologylab.annotation.negation.CompleteNegation", inputCas)
            local event_lst = {}
            local scope_lst = {}
            local xscope_lst = {}
            local focus_lst = {}
            local cue = null
            local negType = null

            if neg["event"] ~= null then
                local tok_idx = 1
                for j, tok in ipairs(neg["event"]) do
                    local begin_idx = tok["begin"]
                    local end_idx = tok["end"]
                    for k, real_tok in ipairs(token_list) do
                        if real_tok:getBegin() == begin_idx and real_tok:getEnd() == end_idx then
                            event_lst[tok_idx] = real_tok
                            tok_idx = tok_idx + 1
                            break
                        end
                    end
                end
            end

            if neg["xscope"] ~= null then
                local tok_idx = 1
                for j, tok in ipairs(neg["xscope"]) do
                    local begin_idx = tok["begin"]
                    local end_idx = tok["end"]
                    for k, real_tok in ipairs(token_list) do
                        if real_tok:getBegin() == begin_idx and real_tok:getEnd() == end_idx then
                            xscope_lst[tok_idx] = real_tok
                            tok_idx = tok_idx + 1
                            break
                        end
                    end

                end
            end

            if neg["scope"] ~= null then
                local tok_idx = 1
                for j, tok in ipairs(neg["scope"]) do
                    local begin_idx = tok["begin"]
                    local end_idx = tok["end"]
                    for k, real_tok in ipairs(token_list) do
                        if real_tok:getBegin() == begin_idx and real_tok:getEnd() == end_idx then
                            scope_lst[tok_idx] = real_tok
                            tok_idx = tok_idx + 1
                            break
                        end
                    end

                end
            end

            if neg["focus"] ~= null then
                local tok_idx = 1
                for j, tok in ipairs(neg["focus"]) do
                    local begin_idx = tok["begin"]
                    local end_idx = tok["end"]
                    for k, real_tok in ipairs(token_list) do
                        if real_tok:getBegin() == begin_idx and real_tok:getEnd() == end_idx then
                            focus_lst[tok_idx] = real_tok
                            tok_idx = tok_idx + 1
                            break
                        end
                    end

                end
            end

            if neg["cue"] ~= null then
                local begin_idx = neg["cue"]["begin"]
                local end_idx = neg["cue"]["end"]
                for k, real_tok in ipairs(token_list) do
                    if real_tok:getBegin() == begin_idx and real_tok:getEnd() == end_idx then
                        cue = real_tok
                        break
                    end
                end
            end

            if cue ~= null then
                neg_obj:setCue(cue)
            end

            if negType ~= null then
                neg_obj:setNegType(negType)
            end

            if #focus_lst ~= 0 then
                local arr = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #focus_lst)
                for j, tok in ipairs(focus_lst) do
                    arr:set(j-1, tok)
                end
                neg_obj:setFocus(arr)
            end

            if #xscope_lst ~= 0 then
                local arr = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #xscope_lst)
                for j, tok in ipairs(xscope_lst) do
                    arr:set(j-1, tok)
                end
                neg_obj:setXscope(arr)
            end

            if #scope_lst ~= 0 then
                local arr = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #scope_lst)
                for j, tok in ipairs(scope_lst) do
                    arr:set(j-1, tok)
                end
                neg_obj:setScope(arr)
            end

            if #event_lst ~= 0 then
                local arr = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #event_lst)
                for j, tok in ipairs(event_lst) do
                    arr:set(j-1, tok)
                end
                neg_obj:setEvent(arr)
            end

            neg_obj:addToIndexes()
            neg_list[neg_counter] = neg_obj
            neg_counter = neg_counter + 1
        end
    end

end
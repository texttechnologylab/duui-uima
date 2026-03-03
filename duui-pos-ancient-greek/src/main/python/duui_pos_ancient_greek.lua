--[[
I same struggled with this Lua file just like with other Lua files. I don't
really know Lua, and the `luajava` bridge feels like dark magic. GitHub Copilot
was actively harmful here. It kept hallucinating Lua/Java syntax that doesn't
actually work. I ended up relying entirely on ChatGPT for debugging and pieced
this together by studying the existing DUUI components (especially the Flair
POS tagger and the Emotion annotator).

Last meaningful edit: Feb 2026
]]

-- Java class bindings --
-- BORROWED. Standard boilerplate from literally every DUUI script.
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- FRAGILE. I wasted an hour trying to `require("json")` because ChatGPT
-- told me to, which broke the whole pipeline. Turns out DUUI injects `json`
-- as a global variable at runtime.

-- SERIALIZE: CAS → JSON request --
function serialize(inputCas, outputStream, parameters)
    -- 1. Extract document text and language
    local doc_text = inputCas:getDocumentText()
    local doc_lang = inputCas:getDocumentLanguage()

    -- I force default to "grc" (Greek) if unspecified, because
    -- sometimes the upstream reader drops the language tag before
    -- the text reaches this component.
    if doc_lang == nil or doc_lang == "x-unspecified" then
        doc_lang = "grc"
    end
    local doc_len = #doc_text

    -- 2. Extract model_name from parameters
    local model_name = nil
    if parameters ~= nil and parameters["model_name"] ~= nil then
        model_name = parameters["model_name"]
    end

    -- 3. Extract existing Sentence annotations
    --
    -- SOLID / CHATGPT. This chunk took three iterations. Originally, I copied
    -- the `JCasUtil:select(inputCas, Sentence):iterator()` pattern from the
    -- Flair POS script. But it threw a massive Java 17 "InaccessibleObjectException"
    -- about ArrayList iterators.
    --
    -- ChatGPT explained that Java 17 blocks reflection on certain native Java
    -- classes, and suggested using UIMA's native index instead of JCasUtil
    -- to bypass the security block. I don't fully grasp UIMA's index internals,
    -- but this approach doesn't crash.
    local sentences = {}
    local sent_counter = 1
    local has_sentences = false
    local sentence_type = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"

    local uimaType = inputCas:getTypeSystem():getType(sentence_type)
    if uimaType ~= nil then
        local sent_index = inputCas:getAnnotationIndex(uimaType)
        if sent_index ~= nil then
            local it = sent_index:iterator()
            while it:hasNext() do
                local sent = it:next()
                sentences[sent_counter] = {
                    begin = sent:getBegin(),
                    ["end"] = sent:getEnd(),
                    text = sent:getCoveredText()
                }
                sent_counter = sent_counter + 1
                has_sentences = true
            end
        end
    end

    -- 4. Build JSON request
    local request = {
        doc_text = doc_text,
        doc_len = doc_len,
        lang = doc_lang,
        model_name = model_name
    }

    if has_sentences then
        request.sentences = sentences
    end

    -- 5. Write to output stream
    outputStream:write(json.encode(request))
end

-- DESERIALIZE: JSON response → CAS annotations --
function deserialize(inputCas, inputStream)
    -- 1. Read and parse the JSON response
    local javaString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- CHATGPT (CRITICAL FIX). The java.lang.String returned above looks
    -- like a string to Lua, but it's actually a Java object reference.
    -- `json.decode` was failing silently and returning nil. ChatGPT caught
    -- this typing mismatch. You *must* cast it to a native Lua string.
    local inputString = tostring(javaString)
    local response = json.decode(inputString)

    if response == nil then
        print("LUA ERROR: json.decode returned nil. Cannot parse response.")
        return
    end

    -- DEBUG PRINT. I added this because I kept getting silent failures
    -- when the Python inference server crashed. This forces Python errors
    -- into the TextImager Java logs.
    if response["errors"] ~= nil and #response["errors"] > 0 then
        for _, err in ipairs(response["errors"]) do
            print("PYTHON API ERROR: " .. tostring(err))
        end
    end

    -- 2. Get type references
    local pos_type = "de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS"

    -- 3. Create POS annotations for each token
    -- BORROWED. The instantiation and `.addToIndexes()` pattern is lifted
    -- almost verbatim from the DUUI Flair POS script.
    if response["tokens"] ~= nil then
        for _, token in ipairs(response["tokens"]) do
            local pos = luajava.newInstance(pos_type, inputCas)
            pos:setBegin(token["begin"])
            pos:setEnd(token["end"])
            pos:setPosValue(token["pos_value"])

            -- Unlike Flair, I'm setting the coarse value too since my Python
            -- script returns it.
            pos:setCoarseValue(token["pos_coarse_value"])
            pos:addToIndexes()
        end
    else
        print("LUA WARNING: 'tokens' array is nil or missing in the response.")
    end

    -- 4. Create MetaData annotation
    -- BORROWED. I took this `DocumentModification` block from the Emotion
    -- and spaCy sentencizer scripts. It leaves an audit trail in the CAS
    -- so that my tags show up properly with a timestamp and model version
    -- in the TextImager UI.
    local meta_type = "org.texttechnologylab.annotation.DocumentModification"
    local meta = luajava.newInstance(meta_type, inputCas)
    meta:setUser(response["model_name"] or "duui-pos-ancient-greek")
    meta:setTimestamp(os.time())
    meta:setComment(
        "POS tagging by " .. (response["model_name"] or "unknown")
        .. " v" .. (response["model_version"] or "0.1.0")
    )
    meta:addToIndexes()

    -- 5. Create AnnotationComment for any errors
    if response["errors"] ~= nil and #response["errors"] > 0 then
        local comment_type = "org.texttechnologylab.annotation.AnnotationComment"
        for _, err in ipairs(response["errors"]) do
            local comment = luajava.newInstance(comment_type, inputCas)
            comment:setKey("error")
            comment:setValue(err)
            comment:addToIndexes()
        end
    end
end
--[[
I do not really know Lua :(. I've never written Lua before this Praktikum.
Copilot is near-useless here because it doesn't understand the luajava
bridge or the DUUI-specific patterns, and keeps hallucinating methods
that don't exist on the Java objects :((. So for this file I leaned heavily
on ChatGPT and on reading existing Lua scripts from other DUUI
annotators in the TTLab repo.

Sources I borrowed from (all from the same GitHub org):
    https://github.com/texttechnologylab/duui-uima
  - duui-transformers-summary/src/.../duui_summary.lua
    (the serialize/deserialize skeleton, the JCasUtil iteration pattern,
    the MetaData and DocumentModification annotation creation)
  - duui-transformers-sentiment/src/.../duui_sentiment.lua
    (the selection-based iteration with Class:forName, the pattern for
    writing results back as typed annotations with begin/end offsets)
  - duui-image-generation/src/.../duui_image_generation.lua
    (the Image annotation type usage, error handling with
    AnnotationComment, writing config key-value pairs back as
    annotation comments. I basically lifted that pattern wholesale)

The structure is always the same across all DUUI Lua scripts.
Once I understood that pattern from reading the existing scripts,
writing this one was mostly a matter of swapping in the right
annotation types and field names for OCR.

ChatGPT wrote the first draft of both functions. I edited field names
and annotation types to match our Python server's request/response
schemas.

Last meaningful edit: Feb 2026
--]]


-- -- Java class bindings --
-- BORROWED. I copied this block from duui-transformers-sentiment and
-- duui-image-generation, then added/removed classes as needed.
-- DUUILuaUtils is a TTLab helper that wraps some common operations
-- like getting document text length (which is apparently not trivial
-- in UIMA because of how surrogate pairs work? I didn't dig into it).
--
-- The string concatenation with ".." is Lua's version of "+" for
-- strings. ChatGPT taught me that. I split long class names across
-- lines because some of these fully-qualified Java names are... something.

StandardCharsets = luajava.bindClass(
    "java.nio.charset.StandardCharsets"
)
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass(
    "org.apache.uima.fit.util.JCasUtil"
)
DUUILuaUtils = luajava.bindClass(
    "org.texttechnologylab.DockerUnifiedUIMAInterface"
        .. ".lua.DUUILuaUtils"
)


-- -- serialize --
-- BORROWED + CHATGPT. The overall skeleton is from duui-transformers-sentiment
-- and duui-image-generation. The Image annotation iteration is adapted
-- from duui-image-generation's deserialize function, but run in
-- reverse. There they *write* Image annotations, here I *read* them.
--
-- ChatGPT wrote the first working version after I described what I
-- needed: "read all Image annotations from the CAS, extract their
-- src/begin/end fields, and send them as a JSON array along with
-- model config parameters."
--
-- I understand the flow: get params, iterate over typed annotations,
-- build a table, encode to JSON. The luajava method-call syntax with
-- the colons (obj:method()) vs dots (obj.field) still trips me up.
-- In Lua, colon means "call this method on the object" and dot means
-- "access this field." I think. ChatGPT explained it three times.

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len =
        DUUILuaUtils:getDocumentTextLength(inputCas)

    local model_name = parameters["model_name"]

    -- Default task to "ocr" if not specified. Most of the time
    -- that's what we want anyway.
    local task = parameters["task"]
    if task == nil then
        task = "ocr"
    end

    -- Cap on how much text the model can generate per image.
    -- 1024 is generous for OCR, a full page of text is usually
    -- well under that in tokens. But better too high than truncated.
    local max_new_tokens = parameters["max_new_tokens"]
    if max_new_tokens == nil then
        max_new_tokens = 1024
    end

    -- -- Collect Image annotations from the CAS --
    -- BORROWED. This pattern is straight from duui-image-generation.
    -- I was going to comment more but then remembered the line from
    -- Game of Throne "You know nothing, Jon Snow." Jon Snow is me,
    -- I am the Jon Snow of Lua.

    local images = {}
    local images_count = 1
    local ImageClass = Class:forName(
        "org.texttechnologylab.annotation.type.Image"
    )
    local images_it =
        JCasUtil:select(inputCas, ImageClass):iterator()

    while images_it:hasNext() do
        local img = images_it:next()
        local image_data = {
            src = img:getSrc(),
            begin = img:getBegin(),
            ["end"] = img:getEnd(),
        }
        images[images_count] = image_data
        images_count = images_count + 1
    end

    outputStream:write(json.encode({
        images = images,
        lang = doc_lang,
        doc_len = doc_len,
        model_name = model_name,
        task = task,
        max_new_tokens = max_new_tokens,
    }))
end


-- -- deserialize --
-- BORROWED + CHATGPT. The overall structure is a patchwork of patterns
-- from the three existing Lua scripts I studied:
--   - Error handling with AnnotationComment: from duui-image-generation
--   - MetaData annotation creation: from duui-transformers-sentiment
--     and duui-transformers-summary (they're basically identical)
--   - Writing results as AnnotationComment key-value pairs: from
--     duui-image-generation's config loop
--
-- ChatGPT helped me stitch these patterns together and adapt them
-- to match the OCRResponse schema from our Python server.
--
-- FRAGILE. This function assumes the Python server's response JSON
-- has exactly the field names we check for. If someone changes the
-- Pydantic model on the Python side without updating this Lua script,
-- results will silently not appear in the CAS.  I don't know how to
-- make this more robust in Lua. There's no schema validation.

function deserialize(inputCas, inputStream)
    -- I would never have figured out this incantation on my own.
    -- Copied verbatim from duui-transformers-summary.
    local inputString = luajava.newInstance(
        "java.lang.String",
        inputStream:readAllBytes(),
        StandardCharsets.UTF_8
    )
    local results = json.decode(inputString)

    -- -- Error handling --
    -- BORROWED from duui-image-generation. Jon Snow speaking here.
    if results["errors"] ~= nil then
        for _, error in ipairs(results["errors"]) do
            local err_annotation = luajava.newInstance(
                "org.texttechnologylab.annotation"
                    .. ".AnnotationComment",
                inputCas
            )
            err_annotation:setKey("error")
            err_annotation:setValue(error)
            err_annotation:addToIndexes()
        end
    end

    -- -- Model metadata --
    -- BORROWED from duui-transformers-sentiment and
    -- duui-transformers-summary.
    if results["model_name"] ~= nil then
        local model_meta = luajava.newInstance(
            "org.texttechnologylab.annotation"
                .. ".model.MetaData",
            inputCas
        )
        model_meta:setModelName(results["model_name"])
        model_meta:setModelVersion(
            results["model_version"]
        )
        model_meta:setSource(results["model_source"])
        model_meta:setLang(results["model_lang"])
        model_meta:addToIndexes()
    end

    -- -- OCR results --
    -- CHATGPT. I asked ChatGPT to write this block.
    -- Prompt was roughly: "iterate over ocr_results from the JSON,
    -- create an AnnotationComment for each, set the key to the task
    -- name and the value to the recognized text."
    --
    -- I'm using AnnotationComment as the output type because our
    -- TypeSystemOCR.xml doesn't define a dedicated OCR annotation
    -- type (yet). AnnotationComment is a generic key-value pair
    -- that's available in the TTLab type system. The key stores
    -- which task produced this result ("ocr", "table", "formula",
    -- etc.) and the value stores the actual text.
    --
    -- REVISIT. This loses the begin/end offset information from
    -- the OCR results. The AnnotationComment gets added at position
    -- 0,0 in the document rather than at the original image's
    -- location. I should probably set the begin/end on the
    -- annotation to match result["begin"] and result["end"], but
    -- I wasn't sure if AnnotationComment supports positional offsets
    -- the way other annotation types do. Need to check the type
    -- system definition. For now the offset data is just... lost
    -- between Python and here. Not great.
    if results["ocr_results"] ~= nil then
        for _, result in ipairs(results["ocr_results"]) do
            local ocr_annotation = luajava.newInstance(
                "org.texttechnologylab.annotation"
                    .. ".AnnotationComment",
                inputCas
            )
            ocr_annotation:setKey(result["task"])
            ocr_annotation:setValue(result["text"])
            ocr_annotation:addToIndexes()
        end
    end

    -- -- Config as annotation comments --
    -- BORROWED + CHATGPT. This pattern is directly from duui-image-generation.
    -- Lua is weakly typed but Java is not, and the luajava bridge
    -- doesn't do implicit conversion. Found that out when it threw
    -- an error on a numeric value. ChatGPT suggested tostring() as the fix.
    if results["config"] ~= nil then
        for key, value in pairs(results["config"]) do
            local config_annotation = luajava.newInstance(
                "org.texttechnologylab.annotation"
                    .. ".AnnotationComment",
                inputCas
            )
            config_annotation:setKey("config_" .. key)
            config_annotation:setValue(tostring(value))
            config_annotation:addToIndexes()
        end
    end
end
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Language = luajava.bindClass("org.texttechnologylab.annotation.Language")
Trans = luajava.bindClass("org.texttechnologylab.annotation.Translation")

function serialize(inputCas, outputStream, parameters)
    local doc_text = inputCas:getDocumentText()
    local model_name = parameters["model_name"]
    local translation_list = parameters["translation"]

    local selections = {}
    local selections_count = 1
    local index_counter = 0
    local languages = JCasUtil:select(inputCas, Language):iterator()
    while languages:hasNext() do
        local language_i = languages:next()
--         print(language_i)
        local text_i = language_i:getCoveredText()
        selections[selections_count] = {}
        selections[selections_count]["text"] = text_i
        selections[selections_count]["begin"] = language_i:getBegin()
        selections[selections_count]["end"] = language_i:getEnd()
        selections[selections_count]["language"] = language_i:getValue()
        index_counter = index_counter + 1
        selections_count = selections_count + 1
    end
    outputStream:write(json.encode({
        selections = selections,
        model_name = model_name,
        translation_list = translation_list,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["begin"] ~= nil then
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]

--         print("meta")
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

--         print("setMetaData")
        local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(model_version)
--         print(model_version)
        model_meta:setModelName(model_name)
--         print(model_name)
        model_meta:setSource(source)
--         print(source)
--         print(model_lang)
        model_meta:addToIndexes()

        local meta = results["meta"]
--         print("meta")
        local begins = results["begin"]
--         print(begins)
        local ends = results["end"]
--         print(ends)
        local arts = results["art"]
--         print(arts)
        local langs = results["languages"]
--         print(langs)
        local translation = results["translation"]
--         print(scores)
        local translation_ref = nil
        for i, begin in ipairs(begins) do
--             print(begin)
            local end_i = ends[i]
--             print(end_i)
            local art_i = arts[i]
--             print(art_i)
            local lang = langs[i]
--             print(lang)
            local translation_i = translation[i]
--             print(translation_i)
            local translation_annotation = luajava.newInstance("org.texttechnologylab.annotation.Translation", inputCas)
            local ref_i = nil
            if art_i == "Normal" then
                ref_i = JCasUtil:selectCovering(inputCas, Language, begin, end_i):iterator():next()
--                 print("Normal")
            else
                ref_i = translation_ref
--                 print("Translated")
            end
--             print(ref_i)
            translation_annotation:setModel(model_meta)
--             print("setModel")
            translation_annotation:setContext(translation_i)
--             print("translation")
            translation_annotation: setReference(ref_i)
--             print("setReference")
            translation_annotation: setValue(lang)
--             print("setValue")
            translation_annotation:addToIndexes()
            translation_ref = translation_annotation
--             print("Language added")
--         print("Languages added")
        end
    end
end

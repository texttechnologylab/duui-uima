StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_text = inputCas:getDocumentText()
    local model_name = parameters["model_name"]
    local selection_types = parameters["selection"]

    local selections = {}
    local selections_count = 1
    for selection_type in string.gmatch(selection_types, "([^,]+)") do
       local sentences = {}
       if selection_type == "text" then
           local s = {
               text = doc_text,
               begin = 0,
               ['end'] = doc_len
           }
           sentences[1] = s
       else
           local sentences_count = 1
           local clazz = Class:forName(selection_type);
           local sentences_it = JCasUtil:select(inputCas, clazz):iterator()
           while sentences_it:hasNext() do
               local sentence = sentences_it:next()
               local s = {
                   text = sentence:getCoveredText(),
                   begin = sentence:getBegin(),
                   ['end'] = sentence:getEnd()
               }
               sentences[sentences_count] = s
               sentences_count = sentences_count + 1
           end
       end

       local selection = {
           sentences = sentences,
           selection = selection_type
       }
       selections[selections_count] = selection
       selections_count = selections_count + 1
    end

    outputStream:write(json.encode({
        selections = selections,
        model_name = model_name
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
        local langs = results["lang"]
--         print(langs)
        local scores = results["scores"]
--         print(scores)
        for i, begin in ipairs(begins) do
--             print("Language")
            local end_i = ends[i]
--             print(end_i)
            local lang = langs[i]
            print(lang)
            local score = scores[i]
            print(score)
            local lang_anno = luajava.newInstance("org.texttechnologylab.annotation.Language", inputCas)
            lang_anno:setBegin(begin)
            lang_anno:setEnd(end_i)
            lang_anno:setValue(lang)
            lang_anno:setScore(score)
            lang_anno:addToIndexes()
            local langmodel = luajava.newInstance("org.texttechnologylab.annotation.LanguageModel", inputCas)
            langmodel:setModel(model_meta)
            langmodel:setLanguage(lang_anno)
            langmodel:addToIndexes()
--             print("Language added")
--         print("Languages added")
        end
    end
end

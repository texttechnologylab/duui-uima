StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
ToxicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = ToxicUtils:getDocumentTextLength(inputCas)

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
        lang = doc_lang,
        doc_len = doc_len,
        model_name = model_name
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["selections"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]

        for i, selection in ipairs(results["selections"]) do
            local selection_type = selection["selection"]
            for j, sentence in ipairs(selection["sentences"]) do
                for k, toxic in ipairs(sentence["toxics"]) do
                    local toxic_anno = luajava.newInstance("org.hucompute.textimager.uima.type.category.CategoryCoveredTagged", inputCas)
                    toxic_anno:setBegin(sentence["sentence"]["begin"])
                    toxic_anno:setEnd(sentence["sentence"]["end"])
                    toxic_anno:setValue(toxic["label"])
                    toxic_anno:setScore(toxic["score"])
                    toxic_anno:addToIndexes()

                    local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
                    meta_anno:setReference(toxic_anno)
                    meta_anno:setName(meta["name"])
                    meta_anno:setVersion(meta["version"])
                    meta_anno:setModelName(meta["modelName"])
                    meta_anno:setModelVersion(meta["modelVersion"])
                    meta_anno:addToIndexes()

                    local meta_selection = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                    meta_selection:setReference(toxic_anno)
                    meta_selection:setKey("selection")
                    meta_selection:setValue(selection_type)
                    meta_selection:addToIndexes()
                end
            end
        end
    end
end

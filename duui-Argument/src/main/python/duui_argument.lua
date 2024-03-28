StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Language = luajava.bindClass("org.texttechnologylab.annotation.Language")
Trans = luajava.bindClass("org.texttechnologylab.annotation.Translation")
Annotation = luajava.bindClass("org.apache.uima.jcas.tcas.Annotation")

function serialize(inputCas, outputStream, parameters)
    local doc_text = inputCas:getDocumentText()
    local model_name = parameters["model_name"]
    local chatgpt_key = parameters["chatgpt_key"]
    local topic = parameters["topic"]
    if chatgpt_key == nil or chatgpt_key == "" then
        chatgpt_key = false
    end
    local selections = {}
    local selections_count = 1
    local index_counter = 0
    local annotations = JCasUtil:select(inputCas, Annotation):iterator()
--     print(annotations)
    while annotations:hasNext() do
        local annotation_i = annotations:next()
--         print(annotation_i)
        local text_i = annotation_i:getCoveredText()
        selections[selections_count] = {}
        selections[selections_count]["text"] = text_i
        selections[selections_count]["begin"] = annotation_i:getBegin()
        selections[selections_count]["end"] = annotation_i:getEnd()
        selections_count = selections_count + 1
    end
    outputStream:write(json.encode({
        selections = selections,
        model_name = model_name,
        chatgpt_key = chatgpt_key,
        topic = topic
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["begin"] ~= nil then
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

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
        model_meta:setLang(model_lang)
        model_meta:addToIndexes()

        local meta = results["meta"]
--         print("meta")
        local begins = results["begin"]
--         print(begins)
        local ends = results["end"]
--         print(ends)
        local keys = results["keys"]
--         print(keys)
        local values = results["values"]
--         print(values)
        local length = results["length"]
--         print(length)
        local topics = results["topics"]
        for index_i, begin in ipairs(begins) do
--             print(begin)
            local end_i = ends[index_i]
--             print(end_i)
            local key_i  = keys[index_i]
--             print(key_i)
            local value_i = values[index_i]
--             print(value_i)
            local len_i = length[index_i]
--             print(len_i)
            local topic_i = topics[index_i]
--             print(topic_i)
            local argument_annotation = luajava.newInstance("org.texttechnologylab.annotation.Argument", inputCas, begin, end_i)
--             print(argument_annotation)
            local fsarray = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, len_i)
--             print(fsarray)
            argument_annotation:setArguments(fsarray)
            argument_annotation:setTopic(topic_i)
            local counter = 0
            for index_j, key_j in ipairs(key_i) do
                local value_j = value_i[index_j]
                argument_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
                argument_i:setReference(argument_annotation)
                argument_i:setKey(key_j)
                argument_i:setValue(value_j)
                argument_i:addToIndexes()
                argument_annotation:setArguments(counter, argument_i)
                counter = counter + 1
            end
            argument_annotation:setModel(model_meta)
            argument_annotation:addToIndexes()
--             print("add")
        end
    end
end

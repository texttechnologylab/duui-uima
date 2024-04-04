StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Language = luajava.bindClass("org.texttechnologylab.annotation.Language")
Trans = luajava.bindClass("org.texttechnologylab.annotation.Translation")
Annotation = luajava.bindClass("org.apache.uima.jcas.tcas.Annotation")
StanceSentence = luajava.bindClass("org.texttechnologylab.annotation.StanceSentence")
Hypothesis = luajava.bindClass("org.texttechnologylab.annotation.Hypothesis")

function serialize(inputCas, outputStream, parameters)
    local doc_text = inputCas:getDocumentText()
    local chatgpt_key = parameters["chatgpt_key"]
    if chatgpt_key == nil or chatgpt_key == "" then
        chatgpt_key = false
    end
    local hypothesis = {}
    local selections_count = 1
    local index_counter = 0
    local annotations = JCasUtil:select(inputCas, Hypothesis):iterator()
--     print(annotations)
    while annotations:hasNext() do
        local annotation_i = annotations:next()
--         print(annotation_i)
        local text_i = annotation_i:getCoveredText()
        local begin_i = annotation_i:getBegin()
        local end_i = annotation_i:getEnd()
        hypothesis[selections_count] = {}
        hypothesis[selections_count]["text"] = text_i
        hypothesis[selections_count]["begin"] = begin_i
        hypothesis[selections_count]["end"] = end_i
        hypothesis[selections_count]["stances"] = {}
        local stance_counter = 1
        local stances = annotation_i:getStances():iterator()
        while stances:hasNext() do
            local stance_i = stances:next()
--             print(stance_i)
            local stance_text = stance_i:getCoveredText()
            local stance_begin = stance_i:getBegin()
            local stance_end = stance_i:getEnd()
            local stance = {}
            hypothesis[selections_count]["stances"][stance_counter] = {}
            hypothesis[selections_count]["stances"][stance_counter]["text"] = stance_text
            hypothesis[selections_count]["stances"][stance_counter]["begin"] = stance_begin
            hypothesis[selections_count]["stances"][stance_counter]["end"] = stance_end
            stance_counter = stance_counter + 1
        end
        selections_count = selections_count + 1
    end
--     print(hypothesis)
--     print("end")
    outputStream:write(json.encode({
        hypothesis = hypothesis,
        chatgpt_key = chatgpt_key
    }))
--     print("send")
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
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
      local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
      meta_anno:setReference(modification_anno)
      meta_anno:setName(meta["name"])
      meta_anno:setVersion(meta["version"])
      meta_anno:setModelName(meta["modelName"])
      meta_anno:setModelVersion(meta["modelVersion"])
      meta_anno:addToIndexes()
--       print("meta")

      local source = results["model_source"]
      local model_version = results["model_version"]
      local model_name = results["model_name"]
      local model_lang = results["model_lang"]
      local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
      model_meta:setModelVersion(model_version)
--       print(model_version)
      model_meta:setModelName(model_name)
--       print(model_name)
      model_meta:setSource(source)
--       print(source)
      model_meta:setLang(model_lang)
      model_meta:addToIndexes()

      local begins = results["begin"]
      local ends = results["end"]
      local begin_stances = results["begin_stances"]
      local end_stances = results["end_stances"]
      local predictions = results["predictions"]
      for i, begin_i in ipairs(begins) do
--             print(begin_i)
            local end_i = ends[i]
--             print(end_i)
            local begin_stance_i = begin_stances[i]
--             print(begin_stance_i)
            local end_stance_i = end_stances[i]
--             print(end_stance_i)
            local prediction_i = predictions[i]
--             print(prediction_i)
            local stance_anno = JCasUtil:selectCovered(inputCas, StanceSentence, begin_stance_i, end_stance_i):iterator():next()
            if meta["modelName"] ~= "gpt3.5" and meta["modelName"] ~= "gpt4" then
                local support = prediction_i["support"]
--                 print(support)
                local oppose = prediction_i["oppose"]
--                 print(oppose)
                local neutral = prediction_i["neutral"]
--                 print(neutral)
                local stance = luajava.newInstance("org.texttechnologylab.annotation.Stance", inputCas, begin_i, end_i)
                stance:setReference(stance_anno)
--                 print(stance)
                stance:setSupport(support)
                stance:setOppose(oppose)
                stance:setNeutral(neutral)
                stance:setModel(model_meta)
                stance:addToIndexes()
            else
                local stance = luajava.newInstance("org.texttechnologylab.annotation.StanceGPT", inputCas, begin_i, end_i)
--                 print(stance)
                stance:setReference(stance_anno)
                local label = prediction_i["label"]
--                 print(label)
                stance:setLabel(label)
                local reason = prediction_i["reason"]
--                 print(reason)
                stance:setReason(reason)
                local confidence = prediction_i["confidence"]
--                 print(confidence)
                stance:setConfidence(confidence)
                stance:setModel(model_meta)
                stance:addToIndexes()
            end
      end
end
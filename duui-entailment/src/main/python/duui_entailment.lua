StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
EntailmentSentence = luajava.bindClass("org.texttechnologylab.annotation.EntailmentSentence")
Entailment = luajava.bindClass("org.texttechnologylab.annotation.Entailment")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_text = inputCas:getDocumentText()
    local chatgpt_key = parameters["chatgpt_key"]
    if chatgpt_key == nil or chatgpt_key == "" then
        chatgpt_key = false
    end
    local entailments = {}
    local counter = 1
    local annotations =  JCasUtil:select(inputCas, EntailmentSentence):iterator()
    while annotations:hasNext() do
        local annotation_i = annotations:next()
        local premise = annotation_i:getPremise()
        local hypothesis = annotation_i:getHypothesis()
        local premise_text = premise:getCoveredText()
--         print(premise_text)
        local hypothesis_text = hypothesis:getCoveredText()
--         print(hypothesis_text)
        entailments[counter] = {}
        entailments[counter]["premise"] = premise_text
        entailments[counter]["hypothesis"] = hypothesis_text
        counter = counter +1
    end

    outputStream:write(json.encode({
        entailments = entailments,
        chatgpt_key = chatgpt_key
    }))
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
-- --       print("meta")

      local source = results["model_source"]
      local model_version = results["model_version"]
      local model_name = results["model_name"]
      local model_lang = results["model_lang"]
      local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
      model_meta:setModelVersion(model_version)
-- --       print(model_version)
      model_meta:setModelName(model_name)
-- --       print(model_name)
      model_meta:setSource(source)
-- --       print(source)
      model_meta:setLang(model_lang)
      model_meta:addToIndexes()

    local predictions = results["predictions"]
    local counter = 1
    local annotations =  JCasUtil:select(inputCas, EntailmentSentence):iterator()
    while annotations:hasNext() do
        local annotation_i = annotations:next()
        local prediction_i = predictions[counter]
        if meta["modelName"] ~= "gpt3.5" and meta["modelName"] ~= "gpt4" then
            local entailment_i =  luajava.newInstance("org.texttechnologylab.annotation.Entailment", inputCas)
            entailment_i:setReference(annotation_i)
--             print(annotation_i:getHypothesis():getCoveredText())
            local entailment = prediction_i["entailment"]
--             print(entailment)
            local non = prediction_i["non_entailment"]
--             print(non)
            entailment_i:setEntailment(entailment)
            entailment_i:setContradiction(non)
            entailment_i:setModel(model_meta)
            entailment_i:addToIndexes()
            counter = counter + 1
        else
            local entailment_i =  luajava.newInstance("org.texttechnologylab.annotation.EntailmentGPT", inputCas)
            entailment_i:setReference(annotation_i)
--             print(annotation_i:getHypothesis():getCoveredText())
            local label = prediction_i["label"]
-- --                 print(label)
            entailment_i:setLabel(label)
            local reason = prediction_i["reason"]
-- --                 print(reason)
            entailment_i:setReason(reason)
            local confidence = prediction_i["confidence"]
            entailment_i:setConfidence(confidence)
            entailment_i:setModel(model_meta)
            entailment_i:addToIndexes()
            counter = counter + 1
        end
    end
end

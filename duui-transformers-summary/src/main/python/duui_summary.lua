StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Annotation = luajava.bindClass("org.apache.uima.jcas.tcas.Annotation")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    -- local doc_text = inputCas:getDocumentText()

    local model_name = parameters["model_name"]
    local summary_length = parameters["summary_length"]
    -- TODO: chunks are not very nice here. (10 sentence / junk)
    -- TODO: while deserialize, add sentences to reconstruct input sentecne and coresponding summary

    local ano_counter = 1
    local all_annotations = {}
    local annos = JCasUtil:select(inputCas, Annotation):iterator()
    while annos:hasNext() do
        local ano = annos:next()
        all_annotations[ano_counter] = {}
        all_annotations[ano_counter]["text"] = ano:getCoveredText()
        all_annotations[ano_counter]["begin"] = ano:getBegin()
        all_annotations[ano_counter]["end"] = ano:getEnd()
        ano_counter = ano_counter + 1
    end
    outputStream:write(json.encode({
        all_annotations = all_annotations,
        lang = doc_lang,
        model_name = model_name,
        summary_length = summary_length
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)


    if results["summaries"] ~= nil and results["meta"] ~= nil then
--         print("GetInfo")
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
--         print(model_lang)
        model_meta:addToIndexes()

        local meta = results["meta"]
--         print("meta")
        local all_begin = results["begin"]
        local all_end = results["end"]
        local summaries = results["summaries"]
        for i, summary in ipairs(summaries) do
            local summary_anno = luajava.newInstance("org.texttechnologylab.annotation.Summary", inputCas)
            local begin_i = all_begin[i]
            local end_i = all_end[i]
            local ano_i = JCasUtil:selectAt(inputCas, Annotation, begin_i, end_i):iterator():next()
            summary_anno:setSummary(summary)
            summary_anno:setReference(ano_i)
            summary_anno:setModel(model_meta)
            summary_anno:addToIndexes()
--             print("Summary added")
        end
--         print("Summaries added")
    end
end

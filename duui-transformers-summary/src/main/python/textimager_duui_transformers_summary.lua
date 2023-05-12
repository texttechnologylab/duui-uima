StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

function serialize(inputCas, outputStream, parameters)
    local doc_lang = inputCas:getDocumentLanguage()
    -- local doc_text = inputCas:getDocumentText()

    local model_name = parameters["model_name"]
    local summary_length = parameters["summary_length"]
    -- TODO: chunks are not very nice here. (10 sentence / junk)
    -- TODO: while deserialize, add sentences to reconstruct input sentecne and coresponding summary

    local sen_counter = 1
    local sent_chunks = {}
    local sents = JCasUtil:select(inputCas, Sentence):iterator()
    while sents:hasNext() do
        local sent = sents:next()
        local chunkidx = math.floor((sen_counter-1) / 10) + 1
        if sent_chunks[chunkidx] == nil then
            sent_chunks[chunkidx] = ""
        end
        sent_chunks[chunkidx] = sent_chunks[chunkidx] .. sent:getCoveredText() .. " "
        sen_counter = sen_counter + 1
    end

    if next(sent_chunks) == nil then
        sent_chunks[1] = "#leer#"
    end

    outputStream:write(json.encode({
        docs = sent_chunks,
        lang = doc_lang,
        model_name = model_name,
        summary_length = summary_length
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)


    if results["summaries"] ~= nil and results["meta"] ~= nil then
        inputCas:createView("SummaryView")
        SummaryView = inputCas:getView("SummaryView")
        SummaryView:setDocumentLanguage(inputCas:getDocumentLanguage())

        summary_text = ""
        for i, summary in ipairs(results["summaries"]) do
            summary_text = summary_text .. " " .. summary

        end
        summary_text = summary_text:sub(2)
        SummaryView:setDocumentText(summary_text)
    end


end

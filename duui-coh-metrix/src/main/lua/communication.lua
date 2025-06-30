StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Token = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Paragraph = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph")

function serialize(inputCas, outputStream, parameters)
    local paragraphs = {}

    local paragraphs_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Paragraph)):listIterator()
    while paragraphs_it:hasNext() do
        local paragraph = paragraphs_it:next()
        local paragraph_data = {
            begin = paragraph:getBegin(),
            ['end'] = paragraph:getEnd(),
            text = paragraph:getCoveredText(),
            sentences = {}
        }
        local sentences_it = luajava.newInstance("java.util.ArrayList", JCasUtil:selectCovered(Sentence, paragraph)):listIterator()
        while sentences_it:hasNext() do
            local sentence = sentences_it:next()
            local sentence_data = {
                begin = sentence:getBegin(),
                ['end'] = sentence:getEnd(),
                text = sentence:getCoveredText(),
                tokens = {}
            }
            local tokens_it = luajava.newInstance("java.util.ArrayList", JCasUtil:selectCovered(Token, sentence)):listIterator()
            while tokens_it:hasNext() do
                local token = tokens_it:next()
                local token_data = {
                    begin = token:getBegin(),
                    ['end'] = token:getEnd(),
                    text = token:getCoveredText(),
                    lemma = token:getLemmaValue(),
                    pos = token:getPosValue(),
                }
                sentence_data.tokens[#sentence_data.tokens + 1] = token_data
            end
            paragraph_data.sentences[#paragraph_data.sentences + 1] = sentence_data
        end
        paragraphs[#paragraphs + 1] = paragraph_data
    end

    outputStream:write(json.encode({
        paragraphs = paragraphs,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    print(results)

--     if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["llm_results"] ~= nil and results["prompts"] ~= nil then
--         local modification_meta = results["modification_meta"]
--         local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
--         modification_anno:setUser(modification_meta["user"])
--         modification_anno:setTimestamp(modification_meta["timestamp"])
--         modification_anno:setComment(modification_meta["comment"])
--         modification_anno:addToIndexes()
--
--     end
end

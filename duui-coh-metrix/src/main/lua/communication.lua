StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Token = luajava.bindClass("org.texttechnologylab.uima.type.spacy.SpacyToken")
NounChunk = luajava.bindClass("org.texttechnologylab.uima.type.spacy.SpacyNounChunk")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Paragraph = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph")
Dependency = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency")

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

                dep_type = ""
                local deps_it = luajava.newInstance("java.util.ArrayList", JCasUtil:selectCovered(Dependency, sentence)):listIterator()
                while deps_it:hasNext() do
                    local dep = deps_it:next()
                    if dep:getDependent() == token then
                        dep_type = dep:getDependencyType()
                        break
                    end
                end

                local vector = nil
                local has_vector = token:getHasVector()
                if has_vector then
                    vector = {}
                    local vector_it = token:getVector():iterator();
                    while vector_it:hasNext() do
                        local vec = vector_it:next()
                        vector[#vector + 1] = vec
                    end
                end

                local token_data = {
                    begin = token:getBegin(),
                    ['end'] = token:getEnd(),
                    text = token:getCoveredText(),
                    lemma = token:getLemmaValue(),
                    pos_value = token:getPos():getPosValue(),
                    pos_coarse = token:getPos():getCoarseValue(),
                    is_alpha = token:getIsAlpha(),
                    is_punct = token:getIsPunct(),
                    dep_type = dep_type,
                    morph_person = token:getMorph():getPerson(),
                    morph_number = token:getMorph():getNumber(),
                    morph_tense = token:getMorph():getTense(),
                    vector = vector,
                    has_vector = has_vector,
                }
                sentence_data.tokens[#sentence_data.tokens + 1] = token_data
            end
            paragraph_data.sentences[#paragraph_data.sentences + 1] = sentence_data
        end
        paragraphs[#paragraphs + 1] = paragraph_data
    end

    local noun_chunks = {}
    local noun_chunks_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, NounChunk)):listIterator()
    while noun_chunks_it:hasNext() do
        local noun_chunk = noun_chunks_it:next()
        local noun_chunk_data = {
            begin = noun_chunk:getBegin(),
            ['end'] = noun_chunk:getEnd(),
        }
        noun_chunks[#noun_chunks + 1] = noun_chunk_data
    end

    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        language = inputCas:getDocumentLanguage(),
        paragraphs = paragraphs,
        noun_chunks = noun_chunks,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["indices"] ~= nil then
        local meta = results["meta"]

        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        for i, index in ipairs(results["indices"]) do
            local index_anno = luajava.newInstance("org.texttechnologylab.uima.type.cohmetrix.Index", inputCas)
            index_anno:setBegin(0)
            index_anno:setEnd(doc_len)
            index_anno:setIndex(index["index"])
            index_anno:setTypeName(index["type_name"])
            index_anno:setLabelTTLab(index["label_ttlab"])
            index_anno:setLabelV3(index["label_v3"])
            index_anno:setLabelV2(index["label_v2"])
            index_anno:setDescription(index["description"])
            index_anno:setValue(index["value"])
            index_anno:setError(index["error"])
            index_anno:setVersion(index["version"])
            index_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(index_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

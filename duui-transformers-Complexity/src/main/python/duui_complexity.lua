StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
complexity = luajava.bindClass("org.texttechnologylab.annotation.Complexity")
SentenceComparison = luajava.bindClass("org.texttechnologylab.annotation.SentenceComparison")
Complex = luajava.bindClass("org.texttechnologylab.annotation.Complexity")
Annotation = luajava.bindClass("org.apache.uima.jcas.tcas.Annotation")
EmbeddingClass = luajava.bindClass("org.texttechnologylab.uima.type.Embedding")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
    local model_name = parameters["model_name"]
--     print(model_name)
    local model_art = parameters["model_art"]
--     print(model_art)
    local complexity_compute = parameters["complexity_compute"]
--     print(complexity_compute)
    local embeddings_keep = parameters["embeddings_keep"]
--     print(embeddings_keep)
    local sentences_i = {}
    local sentences_j = {}
    local selections_count = 1
--     print("h")
    local compares = JCasUtil:select(inputCas, SentenceComparison):iterator()
--     print(compares)
    while compares:hasNext() do
        local compare_i = compares:next()
--         print(compare_i)
        local sentence_i = compare_i:getSentenceI()
--         print(sentence_i)
        local sentence_j = compare_i:getSentenceJ()
--         print(sentence_j)
        sentences_i[selections_count] = {}
        sentences_i[selections_count]["text"] = sentence_i:getCoveredText()
        sentences_i[selections_count]["begin"] = sentence_i:getBegin()
        sentences_i[selections_count]["end"] = sentence_i:getEnd()
--         print("sentences_i")
        sentences_j[selections_count] = {}
        sentences_j[selections_count]["text"] = sentence_j:getCoveredText()
        sentences_j[selections_count]["begin"] = sentence_j:getBegin()
        sentences_j[selections_count]["end"] = sentence_j:getEnd()
        selections_count = selections_count + 1
    end
    outputStream:write(json.encode({
        lang = doc_lang,
        model_name = model_name,
        model_art = model_art,
        complexity_compute = complexity_compute,
        embeddings_keep = embeddings_keep,
        sentences_i = sentences_i,
        sentences_j = sentences_j,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["complexity"] ~= nil then
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
        print(model_version)
        model_meta:setModelName(model_name)
        print(model_name)
        model_meta:setSource(source)
        print(source)
        model_meta:setLang(model_lang)
        print(model_lang)
        model_meta:addToIndexes()

        local meta = results["meta"]
--         print("meta")
        local begin_emb = results["begin_emb"]
--         print(begin_emb)
        local end_emb = results["end_emb"]
--         print("end_emo")
        local embeddings = results["embeddings"]
--         print("results")
        local begin_i = results["begin_i"]
--         print("Len_results")
        local end_i = results["end_i"]
        local begin_j = results["begin_j"]
        local end_j = results["end_j"]
        local complexity = results["complexity"]
        local art = results["art"]
        local len_emb = results["len_emb"]

--         print("set embeddings")
        for index_i, emb in ipairs(embeddings) do
--             print(emb)
--             print(index_i)
            local begin_emb_i = begin_emb[index_i]
--             print(begin_emb_i)
            local end_emb_i = end_emb[index_i]
--             print(end_emb_i)
            local ano_i = JCasUtil:selectAt(inputCas, Annotation, begin_emb_i, end_emb_i):iterator():next()
--             print(ano_i)
            local emb_in = luajava.newInstance("org.texttechnologylab.uima.type.Embedding", inputCas)
--             print(ano_i)
            emb_in:setBegin(begin_emb_i)
            emb_in:setEnd(end_emb_i)
            emb_in:setModelReference(model_meta)
--             print("model_meta")
            local float_array=luajava.newInstance("org.apache.uima.jcas.cas.FloatArray", inputCas, len_emb)
            emb_in:setEmbedding(float_array)
--             print("setEmbedding")
            local counter_i = 0
            for index_emb, emb_i in ipairs(emb) do
                emb_in:setEmbedding(counter_i, emb_i)
                counter_i = counter_i + 1
            end
            emb_in:addToIndexes()
--             print("endEmbedding")
        end
        for index_complex, c_i in ipairs(complexity) do
--             print(c_i)
            local begin_i_complex = begin_i[index_complex]
            local end_i_complex = end_i[index_complex]
            local begin_j_complex = begin_j[index_complex]
            local end_j_complex = end_j[index_complex]
            local art_complex = art[index_complex]
            local ano_complex_i = JCasUtil:selectAt(inputCas, Annotation, begin_i_complex, end_i_complex):iterator():next()
            local ano_complex_j = JCasUtil:selectAt(inputCas, Annotation, begin_j_complex, end_j_complex):iterator():next()
            local complex_ano = luajava.newInstance("org.texttechnologylab.annotation.Complexity", inputCas)
            complex_ano:setSentenceI(ano_complex_i)
            complex_ano:setSentenceJ(ano_complex_j)
            complex_ano:setKind(art_complex)
            complex_ano:setOutput(c_i)
            complex_ano:setModel(model_meta)
            complex_ano:addToIndexes()
        end
    end
--     print("end")
end

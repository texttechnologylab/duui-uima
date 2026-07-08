StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
FloatArray = luajava.bindClass("org.apache.uima.jcas.cas.FloatArray")

function serialize(inputCas, outputStream, parameters)
    local batch_size = parameters["batch_size"]
    local model_name = parameters["model_name"]

    local sentences = {}
    local sentences_count = 1
    local sentences_it = JCasUtil:select(inputCas, Sentence):iterator()
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

    outputStream:write(json.encode({
        sentences = sentences,
        model_name = model_name,
        batch_size = batch_size
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["embeddings"] ~= nil then
        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]

        local meta_data_anno = luajava.newInstance("org.texttechnologylab.annotation.MetaData", inputCas)
        meta_data_anno:setSource(meta["modelName"])
        meta_data_anno:addToIndexes()

        for i, embedding in ipairs(results["embeddings"]) do
            local embedding_anno = luajava.newInstance("org.texttechnologylab.uima.type.Embedding", inputCas)
            embedding_anno:setBegin(embedding['begin'])
            embedding_anno:setEnd(embedding['end'])
            local vector_array = FloatArray:create(inputCas, embedding["vector"])
            embedding_anno:setEmbedding(vector_array)
            embedding_anno:setModelReference(meta_data_anno)
            embedding_anno:addToIndexes()

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(embedding_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

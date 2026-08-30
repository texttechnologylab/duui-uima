StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Embedding = luajava.bindClass("org.texttechnologylab.uima.type.Embedding")
DocumentMetaData = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")

function serialize(inputCas, outputStream, parameters)
    local db_backend = parameters["db_backend"]
    local target_table = parameters["target_table"]
    local target_table_prefix = parameters["target_table_prefix"]

    -- DocumentMetaData ist optional: laeuft das Tool ausserhalb einer Pipeline,
    -- die eine Dokument-ID setzt, wird stattdessen "unknown_document" verwendet.
    local doc_id = "unknown_document"
    local meta_it = JCasUtil:select(inputCas, DocumentMetaData):iterator()
    if meta_it:hasNext() then
        doc_id = meta_it:next():getDocumentId()
    end

    local embeddings = {}
    local count = 1
    local embedding_it = JCasUtil:select(inputCas, Embedding):iterator()
    while embedding_it:hasNext() do
        local embedding = embedding_it:next()

        local model_name = "unknown_model"
        local model_ref = embedding:getModelReference()
        if model_ref ~= nil then
            model_name = model_ref:getSource()
        end

        local vector = {}
        local values = embedding:getEmbedding()
        for i = 0, values:size() - 1 do
            vector[i + 1] = values:get(i)
        end

        embeddings[count] = {
            begin = embedding:getBegin(),
            ['end'] = embedding:getEnd(),
            vector = vector,
            model_name = model_name
        }
        count = count + 1
    end

    outputStream:write(json.encode({
        doc_id = doc_id,
        db_backend = db_backend,
        target_table = target_table,
        target_table_prefix = target_table_prefix,
        embeddings = embeddings
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local result = json.decode(inputString)

    if result["status"] ~= nil then
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser("duui-vector-db-writer")
        modification_anno:setTimestamp(result["timestamp"])
        modification_anno:setComment(result["comment"])
        modification_anno:addToIndexes()
    end
end

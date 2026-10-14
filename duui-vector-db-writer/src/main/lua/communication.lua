StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Embedding = luajava.bindClass("org.texttechnologylab.uima.type.Embedding")
DocumentMetaData = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")

DUUI_DEFAULT_DOCUMENT_ID = "UIMA-Document"

-- Werden unveraendert an den Service durchgereicht. Nicht gesetzte Parameter
-- sind nil und fehlen damit im JSON -- der Service nimmt dann den Default.
PASSTHROUGH_PARAMETERS = {
    "db_backend",
    "target_table",
    "target_table_prefix",
    "fail_on_error",
    -- Postgres-Verbindung
    "pg_connection_string",
    "pg_host",
    "pg_port",
    "pg_database",
    "pg_user",
    "pg_password",
    -- Qdrant-Verbindung
    "qdrant_url",
    "qdrant_host",
    "qdrant_port",
    "qdrant_api_key",
    "qdrant_distance"
}

function serialize(inputCas, outputStream, parameters)
    -- Die Dokument-ID ist Teil des Primaerschluessels; ohne sie wuerden sich
    -- Dokumente gegenseitig ueberschreiben. Wie DocumentMetaData.get() in
    -- DKPro schlaegt das Dokument deshalb fehl, statt einen Platzhalter zu nehmen.
    local doc_id = nil
    local meta_it = JCasUtil:select(inputCas, DocumentMetaData):iterator()
    if meta_it:hasNext() then
        doc_id = meta_it:next():getDocumentId()
    end
    if doc_id == nil or doc_id == "" then
        error("duui-vector-db-writer: CAS has no DocumentMetaData with a documentId")
    end
    -- DUUIComposer.run(JCas) legt fuer einen CAS ohne DocumentMetaData selbst
    -- eine mit dieser ID an -- alle solchen Dokumente haetten dieselbe ID und
    -- wuerden sich gegenseitig ueberschreiben.
    if doc_id == DUUI_DEFAULT_DOCUMENT_ID then
        error("duui-vector-db-writer: documentId is DUUI's default \"" .. DUUI_DEFAULT_DOCUMENT_ID
                .. "\" (CAS had no DocumentMetaData); set a unique documentId per document")
    end

    local embeddings = {}
    local count = 1
    local embedding_it = JCasUtil:select(inputCas, Embedding):iterator()
    while embedding_it:hasNext() do
        local embedding = embedding_it:next()

        -- Modellname bestimmt Tabelle/Collection und ist Teil des Schluessels.
        local model_ref = embedding:getModelReference()
        local model_name = nil
        if model_ref ~= nil then
            model_name = model_ref:getSource()
        end
        if model_name == nil or model_name == "" then
            error("duui-vector-db-writer: Embedding " .. embedding:getBegin() .. "-" .. embedding:getEnd()
                    .. " in document " .. doc_id .. " has no modelReference with a source (model name)")
        end

        local vector = {}
        local values = embedding:getEmbedding()
        if values ~= nil then
            for i = 0, values:size() - 1 do
                vector[i + 1] = values:get(i)
            end
        end

        embeddings[count] = {
            begin = embedding:getBegin(),
            ['end'] = embedding:getEnd(),
            vector = vector,
            model_name = model_name
        }
        count = count + 1
    end

    local request = {
        doc_id = doc_id,
        embeddings = embeddings
    }
    for _, name in ipairs(PASSTHROUGH_PARAMETERS) do
        request[name] = parameters[name]
    end

    outputStream:write(json.encode(request))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local result = json.decode(inputString)

    -- Bei fail_on_error=true kommen Fehler gar nicht hier an (HTTP-Fehler,
    -- DUUI bricht ab); bei false wird der Fehler hier protokolliert.
    if result["modification_meta"] ~= nil then
        local modification_meta = result["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()
    end
end

---Indicates that this component supports the "new" `process` method.
SUPPORTS_PROCESS = true
---Indicates that this component does NOT support the old `serialize`/`deserialize` methods.
SUPPORTS_SERIALIZE = false

------------------------------------------------------

---Process the sentences in the given JCas in small batches.
---@param sourceCas any JCas (view) to process
---@param handler any RequestHandler with a connection to the component service
---@param parameters table optional parameters
---@param targetCas any JCas (view) to write the results to (optional)
function process(sourceCas, handler, parameters, targetCas)
    parameters = parameters or {}
    local language = parameters.language_override or sourceCas:getDocumentLanguage()
    if language == nil or language == "" or language == "x-unspecified" then
        language = "xx"
    end
    local config = {
        spacy_language = language
    }
    targetCas = targetCas or sourceCas

    ---Construct a request and send it to the component's /v1/process endpoint
    local response = handler:process(
        json.encode({
            text = sourceCas:getDocumentText(),
            config = config,
        })
    )

    ---Check if the response is valid, otherwise handle the error
    if not response:ok() then
        error("Error " .. response:statusCode() .. " in communication with component: " .. response:body())
    end

    local results = json.decode(response:bodyUtf8())

    ---Collect references to all created annotations in a table for later use in metadata
    ---@type table<integer, any>
    local references = {}

    ---Create a Sentence annotation for each detected sentence
    for _, sentence in ipairs(results.sentences) do
        local annotation = luajava.newInstance("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", targetCas)
        annotation:setBegin(sentence["begin"])
        annotation:setEnd(sentence["end"])
        annotation:addToIndexes()

        references[#references + 1] = annotation
    end

    ---After successful processing, create a SpacyAnnotatorMetaData annotation

    ---The metadata is provided by the component in the response
    ---@type table<string, any>
    local metadata = results.metadata

    local reference_array = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", targetCas, #references)
    for i, ref in ipairs(references) do
        reference_array:set(i - 1, ref)
    end

    local annotation = luajava.newInstance("org.texttechnologylab.annotation.SpacyAnnotatorMetaData", targetCas)
    annotation:setReference(reference_array)
    annotation:setName(metadata.name)
    annotation:setVersion(metadata.version)
    annotation:setSpacyVersion(metadata.spacy_version)
    annotation:setModelName(metadata.model_name)
    annotation:setModelVersion(metadata.model_version)
    annotation:setModelLang(metadata.model_lang)
    annotation:setModelSpacyVersion(metadata.model_spacy_version)
    annotation:setModelSpacyGitVersion(metadata.model_spacy_git_version)
    annotation:addToIndexes()
end

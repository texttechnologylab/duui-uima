---Indicates that this component supports the "new" `process` method.
SUPPORTS_PROCESS = true
---Indicates that this component does NOT support the old `serialize`/`deserialize` methods.
SUPPORTS_SERIALIZE = false

------------------------------------------------------

local FSArray = "org.apache.uima.jcas.cas.FSArray"
local DocumentModification = "org.texttechnologylab.annotation.DocumentModification"
local Taxon = "org.texttechnologylab.annotation.biofid.gnfinder.Taxon"
local VerifiedTaxon = "org.texttechnologylab.annotation.biofid.gnfinder.VerifiedTaxon"
local OddsDetails = "org.texttechnologylab.annotation.biofid.gnfinder.OddsDetails"
local MetaData = "org.texttechnologylab.annotation.biofid.gnfinder.MetaData"
local MetaDataKeyValue = "org.texttechnologylab.annotation.biofid.gnfinder.MetaDataKeyValue"


---Set fields common to Taxon and VerifiedTaxon
---@param targetCas any the target JCas
---@param taxon any the Taxon or VerifiedTaxon object
---@param name table<string, any> the name object from the GNFinder response
local function handle_common_taxon_fields(targetCas, taxon, name)
    taxon:setBegin(name.start)
    taxon:setEnd(name.start + string.len(name.verbatim))
    taxon:setValue(name.name)
    taxon:setCardinality(name.cardinality)
    taxon:setOddsLog10(name.oddsLog10)
    if name.oddsDetails ~= nil then
        local odds_details = luajava.newInstance(FSArray, targetCas, #name.oddsDetails)
        local details = nil
        for i, detail in ipairs(name.oddsDetails) do
            details = luajava.newInstance(OddsDetails, targetCas)
            details:setFeature(detail.feature)
            details:setOdds(detail.value)
            details:addToIndexes()
            odds_details:set(i - 1, details)
        end
        odds_details:addToIndexes()
        taxon:setOddsDetails(odds_details)
    end
end

---Decode a string of sources in to a table of integers
---@param sources string string to decode
---@return table<integer, integer> table of integers
local function decode_sources(sources)
    local decoded = json.decode(sources)
    if type(decoded) == "table" then
        return decoded
    elseif type(decoded) == "number" then
        return { decoded }
    end
    error("Invalid sources format '" .. sources .. "', expected a json array or a number")
end

---Handle the response from GNFinder and create the corresponding annotations
---@param targetCas any the target JCas
---@param gnfinder_names table<string, any> the names found by GNFinder
---@param gnfinder_metadata table<string, string> the metadata from the GNFinder response
local function handle_response(targetCas, gnfinder_names, gnfinder_metadata)
    local references = {}
    for _, name in ipairs(gnfinder_names) do
        if name.verification == nil then
            local taxon = luajava.newInstance(Taxon, targetCas)

            handle_common_taxon_fields(targetCas, taxon, name)

            taxon:addToIndexes()
            references[#references + 1] = taxon
        else
            local verified_names = {}
            if name.verification.bestResult ~= nil then
                verified_names[1] = name.verification.bestResult
            elseif name.verification.results ~= nil then
                verified_names = name.verification.results
            else
                ---unreachable
                error("Invalid response format: response must contain either 'bestResult' or 'results'!")
            end

            for _, verif in ipairs(verified_names) do
                local taxon = luajava.newInstance(VerifiedTaxon, targetCas)

                handle_common_taxon_fields(targetCas, taxon, name)

                taxon:setDataSourceId(verif.dataSourceId)
                taxon:setRecordId(verif.recordId)

                if verif.globalId ~= nil then
                    taxon:setGlobalId(verif.globalId)
                end
                if verif.localId ~= nil then
                    taxon:setLocalId(verif.localId)
                end
                if verif.outlink ~= nil then
                    taxon:setOutlink(verif.outlink)
                    taxon:setIdentifier(verif.outlink)
                else
                    taxon:setIdentifier(verif.recordId)
                end
                taxon:setSortScore(verif.sortScore)
                taxon:setMatchedName(verif.matchedName)
                taxon:setCurrentName(verif.currentName)
                taxon:setMatchedCanonicalSimple(verif.matchedCanonicalSimple)
                taxon:setMatchedCanonicalFull(verif.matchedCanonicalFull)
                taxon:setTaxonomicStatus(verif.taxonomicStatus)
                taxon:setMatchType(verif.matchType)
                taxon:setEditDistance(verif.editDistance)

                taxon:addToIndexes()
                references[#references + 1] = taxon
            end
        end
    end

    ---GNFinder metadata handling
    local metadata = luajava.newInstance(MetaData, targetCas)

    ---@type table<string, any> GNFinder metadata
    metadata:setDate(gnfinder_metadata.date)
    metadata:setVersion(gnfinder_metadata.gnfinderVersion)
    metadata:setLanguage(gnfinder_metadata.language)

    ---Add references to all created taxon annotations

    local taxon_references = luajava.newInstance(FSArray, targetCas, #references)
    for i, ref in ipairs(references) do
        taxon_references:set(i - 1, ref)
    end
    taxon_references:addToIndexes()
    metadata:setReferences(taxon_references)

    ---Add metadata fields for any other settings (starting with "with"), like "withNoBayes"

    local other = {}
    for key, value in pairs(gnfinder_metadata) do
        if string.sub(key, 1, 4) == "with" then
            other[#other + 1] = { key, value }
        end
    end

    if #other > 0 then
        local fs_array = luajava.newInstance(FSArray, targetCas, #other)
        for i, kv in ipairs(other) do
            local key, value = table.unpack(kv)
            local fs = luajava.newInstance(MetaDataKeyValue, targetCas)
            fs:setKey(key)
            fs:setValue(value)
            fs:addToIndexes()
            fs_array:set(i - 1, fs)
        end
        fs_array:addToIndexes()
        metadata:setOther(fs_array)
    end

    metadata:addToIndexes()

    -- Add modification annotation
    local document_modification = luajava.newInstance(DocumentModification, targetCas)
    document_modification:setUser("duui-lite-gnfinder")
    document_modification:setTimestamp(gnfinder_metadata.date)
    document_modification:setComment(
        "GNFinder " .. gnfinder_metadata.gnfinderVersion ..
        ", language = " .. gnfinder_metadata.language
    )
    document_modification:addToIndexes()
end

---Process the document text of the given source JCas
---@param sourceCas any JCas (view) to process
---@param handler any DuuiHttpRequestHandler with a connection to the running component
---@param parameters table<string, string> optional parameters
---@param targetCas any JCas (view) to write the results to (optional)
function process(sourceCas, handler, parameters, targetCas)
    targetCas = targetCas or sourceCas

    ---@type table <string, any> POST query for the GNFinder HTTP API
    local query = {
        text = sourceCas:getDocumentText(),
        language = parameters.language or "detect",
        ambiguousNames = parameters.ambiguousNames == "true",
        noBayes = parameters.noBayes == "true",
        oddsDetails = parameters.oddsDetails == "true",
        verification = parameters.verification ~= "false",
        sources = decode_sources(parameters.sources or "11"),
        allMatches = parameters.allMatches == "true",
        ---Format is fixed to "compact" (JSON)
        format = "compact",
    }

    local lang = sourceCas:getDocumentLanguage()
    if lang ~= nil then
        if lang:lower():sub(2) == "de" then
            query["language"] = "deu"
        elseif lang:lower():sub(2) then
            query["language"] = "eng"
        end
    end

    handler:setHeader("Content-Type", "application/json")
    local response = handler:post("/api/v1/find", json.encode(query))

    if not response:ok() then
        error("Error " .. response:statusCode() .. " in communication with component: " .. response:bodyUtf8())
    end

    ---@type table<string, table>
    local results = json.decode(response:bodyUtf8())

    handle_response(targetCas, results.names, results.metadata)
end

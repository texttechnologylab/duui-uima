-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

Taxon = "org.texttechnologylab.annotation.biofid.gnfinder.Taxon"
VerifiedTaxon = "org.texttechnologylab.annotation.biofid.gnfinder.VerifiedTaxon"
OddsDetails = "org.texttechnologylab.annotation.biofid.gnfinder.OddsDetails"
MetaData = "org.texttechnologylab.annotation.biofid.gnfinder.MetaData"
MetaDataKeyValue = "org.texttechnologylab.annotation.biofid.gnfinder.MetaDataKeyValue"

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
--  - parameters: Table/Dictonary of parameters that should be used to configure the annotator
function serialize(inputCas, outputStream, parameters)
    local params = {
        text = inputCas:getDocumentText(),
        language = parameters.language or "detect",
        ambiguousNames = parameters.ambiguousNames == "true",
        noBayes = parameters.noBayes == "true",
        oddsDetails = parameters.oddsDetails == "true",
        verification = parameters.verification ~= "false",
        sources = parameters.sources or { 11 },
        allMatches = parameters.allMatches == "true",
    }

    local lang = inputCas:getDocumentLanguage()
    if lang ~= nil then
        if lang:lower():sub(2) == "de" then
            params["language"] = "deu"
        elseif lang:lower():sub(2) then
            params["language"] = "eng"
        end
    end

    outputStream:write(json.encode(params))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    -- Get string from stream, assume UTF-8 encoding
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)

    -- Parse JSON data from string into object
    local results = json.decode(inputString)

    -- GNFinder recognition result handling
    local taxon_anno = nil
    local taxon_references = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #results.results)
    for i, taxon in ipairs(results.results) do
        if taxon.recordId ~= nil then
            taxon_anno = luajava.newInstance(VerifiedTaxon, inputCas)
        else
            taxon_anno = luajava.newInstance(Taxon, inputCas)
        end

        taxon_anno:setBegin(taxon.begin)
        taxon_anno:setEnd(taxon["end"])
        taxon_anno:setBegin(taxon.begin)
        taxon_anno:setValue(taxon.value)
        taxon_anno:setIdentifier(taxon.identifier)
        taxon_anno:setCardinality(taxon.cardinality)
        taxon_anno:setOddsLog10(taxon.oddsLog10)
        if taxon.oddsDetails ~= nil then
            local odds_details = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #taxon.oddsDetails)
            local details_anno = nil
            for i, details in ipairs(taxon.oddsDetails) do
                details_anno = luajava.newInstance(OddsDetails, inputCas)
                details_anno:setFeature(details.feature)
                details_anno:setValue(details.value)
                details_anno:addToIndexes()
                odds_details:set(i - 1, details_anno)
            end
            odds_details:addToIndexes()
            taxon_anno:setOddsDetails(odds_details)
        end

        if taxon.recordId ~= nil then
            taxon_anno:setDataSourceId(taxon.dataSourceId)
            taxon_anno:setRecordId(taxon.recordId)

            if taxon.globalId ~= nil then
                taxon_anno:setGlobalId(taxon.globalId)
            end
            if taxon.localId ~= nil then
                taxon_anno:setLocalId(taxon.localId)
            end
            if taxon.outlink ~= nil then
                taxon_anno:setOutlink(taxon.outlink)
            end
            taxon_anno:setSortScore(taxon.sortScore)
            taxon_anno:setMatchedName(taxon.matchedName)
            taxon_anno:setCurrentName(taxon.currentName)
            taxon_anno:setMatchedCanonicalSimple(taxon.matchedCanonicalSimple)
            taxon_anno:setMatchedCanonicalFull(taxon.matchedCanonicalFull)
            taxon_anno:setTaxonomicStatus(taxon.taxonomicStatus)
            taxon_anno:setMatchType(taxon.matchType)
            taxon_anno:setEditDistance(taxon.editDistance)
        end

        taxon_anno:addToIndexes()
        taxon_references:set(i - 1, taxon_anno)
    end
    taxon_references:addToIndexes()

    -- GNFinder metadata handling
    local metadata_anno = luajava.newInstance(MetaData, inputCas)

    ---@type table
    local metadata = results.metadata
    metadata_anno:setDate(metadata.date)
    metadata_anno:setVersion(metadata.version)
    metadata_anno:setLanguage(metadata.language)
    metadata_anno:setReferences(taxon_references)

    if metadata.other ~= nil and #metadata.other > 0 then
        local fs_array = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #metadata.other)
        for i, kv in ipairs(metadata.other) do
            local key, value = table.unpack(kv)
            local fs = luajava.newInstance(MetaDataKeyValue, inputCas)
            fs:setKey(key)
            fs:setValue(value)
            fs:addToIndexes()
            fs_array:set(i - 1, fs)
        end
        fs_array:addToIndexes()
        metadata_anno:setOther(fs_array)
    end

    metadata_anno:addToIndexes()

    -- Add modification annotation
    local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
    modification_anno:setUser("GNFinder v2")
    modification_anno:setTimestamp(metadata.date)
    modification_anno:setComment("GNFinder, " .. metadata.version .. ", language = " .. metadata.language)
    modification_anno:addToIndexes()
end

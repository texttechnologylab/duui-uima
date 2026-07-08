-- Bind static classes from Java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

---This component does NOT support the new `process` method.
SUPPORTS_PROCESS = false

---This component supports the old `serialize`/`deserialize` methods.
SUPPORTS_SERIALIZE = true

ANNOTATION_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.ner.type.Location"

---Serialize annotations from the DUUI source view.
---DUUI resolves `.withSourceView(...)` before calling this function.
---Therefore inputCas is already the source view.
---@param inputCas any source JCas view
---@param outputStream any output stream to the remote component
---@param parameters table optional parameters
function serialize(inputCas, outputStream, parameters)
    parameters = parameters or {}

    local annotation_type = parameters.annotation_type or ANNOTATION_TYPE

    local query = {
        mode = parameters.mode or "find",
        result_selection = parameters.result_selection or "first",
        queries = {}
    }

    if parameters.filter ~= nil then
        query.filter = parameters.filter
    end

    if query.mode ~= "find" and parameters.max_dist ~= nil then
        query.max_dist = tostring(parameters.max_dist)
    end

    if query.mode == "levenshtein" and parameters.state_limit ~= nil then
        query.state_limit = tostring(parameters.state_limit)
    end

    if parameters.min_length ~= nil then
        query.min_length = tostring(parameters.min_length)
    end

    local annotation_class = luajava.bindClass(annotation_type)
    local iterator = JCasUtil:select(inputCas, annotation_class):iterator()

    local index = 1

    while iterator:hasNext() do
        local entity = iterator:next()

        local begin_pos = entity:getBegin()
        local end_pos = entity:getEnd()

        local ok_text, text = pcall(function()
            return entity:getCoveredText()
        end)

        if not ok_text then
            print("GeoNamesFST WARN: getCoveredText failed for " ..
                tostring(begin_pos) .. "-" .. tostring(end_pos) ..
                ": " .. tostring(text))
            text = nil
        end

        if text == nil or text == "" then
            local ok_view, fallbackText = pcall(function()
                local cas = inputCas:getCas()
                local initialCas = cas:getView("_InitialView")
                local docText = initialCas:getDocumentText()

                if docText ~= nil then
                    return docText:substring(begin_pos, end_pos)
                end

                return nil
            end)

            if ok_view and fallbackText ~= nil and fallbackText ~= "" then
                text = fallbackText
            else
                print("GeoNamesFST WARN: InitialView fallback failed for " ..
                    tostring(begin_pos) .. "-" .. tostring(end_pos) ..
                    ": " .. tostring(fallbackText))
            end
        end

        if text ~= nil and text ~= "" then
            query.queries[#query.queries + 1] = {
                reference = tostring(index),
                text = text,
                begin = begin_pos,
                ["end"] = end_pos
            }

            index = index + 1
        else
            print("GeoNamesFST SKIP: no text for annotation " ..
                tostring(begin_pos) .. "-" .. tostring(end_pos))
        end
    end

    outputStream:write(json.encode(query))
end

---Deserialize GeoNames results into the DUUI target view.
---DUUI resolves/creates `.withTargetView(...)` before calling this function.
---Therefore inputCas is already the target view.
---@param inputCas any target JCas view
---@param inputStream any response stream from the remote component
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance(
        "java.lang.String",
        inputStream:readAllBytes(),
        StandardCharsets.UTF_8
    )

    local results = json.decode(inputString)

    if results == nil then
        return
    end

    if results.results ~= nil then
        for _, entity in ipairs(results.results) do
            add_geonames_annotation(inputCas, entity)
        end
    end

    if results.modification ~= nil then
        add_document_modification(inputCas, results.modification)
    end
end

---Create one GeoNamesEntity annotation in the target view.
---@param targetCas any target JCas view
---@param entity table one result entry from the component
function add_geonames_annotation(targetCas, entity)
    if entity == nil or entity.entry == nil then
        return
    end

    local gn = entity.entry

    local begin_pos = tonumber(entity.begin)
    local end_pos = tonumber(entity["end"])

    if begin_pos == nil or end_pos == nil then
        error(
            "Missing begin/end offsets in GeoNames response for reference: " ..
            tostring(entity.reference)
        )
    end

    local annotation = luajava.newInstance(
        "org.texttechnologylab.annotation.geonames.GeoNamesEntity",
        targetCas
    )

    annotation:setBegin(begin_pos)
    annotation:setEnd(end_pos)

    if gn.id ~= nil then
        annotation:setId(tonumber(gn.id))
    end

    if gn.name ~= nil then
        annotation:setName(gn.name)
    end

    if gn.latitude ~= nil then
        annotation:setLatitude(gn.latitude)
    end

    if gn.longitude ~= nil then
        annotation:setLongitude(gn.longitude)
    end

    if gn.feature_class ~= nil then
        annotation:setFeatureClass(gn.feature_class)
    end

    if gn.feature_code ~= nil then
        annotation:setFeatureCode(gn.feature_code)
    end

    if gn.country_code ~= nil then
        annotation:setCountryCode(gn.country_code)
    end

    if gn.adm1 ~= nil then
        annotation:setAdm1(gn.adm1)
    end

    if gn.adm2 ~= nil then
        annotation:setAdm2(gn.adm2)
    end

    if gn.adm3 ~= nil then
        annotation:setAdm3(gn.adm3)
    end

    if gn.adm4 ~= nil then
        annotation:setAdm4(gn.adm4)
    end

    if gn.elevation ~= nil then
        annotation:setElevation(gn.elevation)
    end

    annotation:addToIndexes()
end

---Add DocumentModification annotation to the target view.
---@param targetCas any target JCas view
---@param modification table modification metadata from component
function add_document_modification(targetCas, modification)
    if modification == nil then
        return
    end

    local document_modification = luajava.newInstance(
        "org.texttechnologylab.annotation.DocumentModification",
        targetCas
    )

    if modification.user ~= nil then
        document_modification:setUser(modification.user)
    end

    if modification.timestamp ~= nil then
        document_modification:setTimestamp(modification.timestamp)
    end

    if modification.comment ~= nil then
        document_modification:setComment(modification.comment)
    end

    document_modification:addToIndexes()
end

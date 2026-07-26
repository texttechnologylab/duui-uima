-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    local doc_text = inputCas:getDocumentText()
    local linking = "gbif_backbone"
    local threshold = 0.7
    if params["linking"] ~= nil then
        linking = params["linking"]
    end
    if params["threshold"] ~= nil then
        threshold = params["threshold"]
    end

    print(linking)
    print(doc_text)
    print(threshold)

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        text = doc_text,
        linking = linking,
        threshold = threshold
    }))
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

    for i, tax in ipairs(results["taxons"]) do

        local taxon = luajava.newInstance("org.texttechnologylab.annotation.type.Taxon", inputCas)
        taxon:setBegin(tax["begin"])
        taxon:setEnd(tax["end"])
        taxon:addToIndexes()

        for c, comment in ipairs(tax["comment"]) do

            local cID = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            cID:setKey("linking")
            cID:setValue(comment["id"])
            cID:setReference(taxon)
            cID:addToIndexes()

            local cValue = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            cValue:setKey("value")
            cValue:setValue(comment["value"])
            cValue:setReference(cID)
            cValue:addToIndexes()

            local cPropability = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            cPropability:setKey("propability")
            cPropability:setValue(comment["propability"])
            cPropability:setReference(cID)
            cPropability:addToIndexes()


        end

    end

end

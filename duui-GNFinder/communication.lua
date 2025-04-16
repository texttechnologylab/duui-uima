serial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasSerializer")
deserial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasDeserializer")
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, params)
    local parameterText = " "

    if(params["adjustOdds"] ~= nil and params["adjustOdds"] == 'true') then
        parameterText = parameterText .. "-a "
    end
    if(params["allMatches"] ~= nil and params["allMatches"] == 'true') then
        parameterText = parameterText .. "-M "
    end
    if(params["ambiguousUninomials"] ~= nil and params["ambiguousUninomials"] == 'true') then
        parameterText = parameterText .. "-A "
    end

    if(params["detailsOdds"] ~= nil and params["detailsOdds"] == 'true') then
        parameterText = parameterText .. "-d "
    end
    if(params["lang"] ~= nil and params["lang"] ~= '') then
        parameterText = parameterText .. "-l " .. params["lang"] .. " "
    end
    if(params["noBayes"] ~= nil and params["noBayes"] == 'true') then
        parameterText = parameterText .. "-n "
    end
    if(params["sources"] ~= nil and params["sources"] ~= '') then
        parameterText = parameterText .. "-s " .. params["sources"] .. " "
    end

    if(params["uniqueNames"] ~= nil and params["uniqueNames"] == 'true') then
        parameterText = parameterText .. "-u "
    end

    -- Verify is on by default
    if(not (params["verify"] ~= nil and params["verify"] == 'false')) then
        parameterText = parameterText .. "-v "
    end


    outputStream:write(json.encode({args = parameterText}))
    serial:serialize(inputCas:getCas(),outputStream)
end

function deserialize(inputCas, inputStream)
  inputCas:reset()
  deserial:deserialize(inputStream,inputCas:getCas(),true)
end

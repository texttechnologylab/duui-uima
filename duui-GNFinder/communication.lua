serial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasSerializer")
deserial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasDeserializer")
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

function serialize(inputCas, outputStream, params)
    local parameterText = " "
    local param = params["adjustOdds"]

    if(true) then
        parameterText = parameterText .. "-a "
    end
    if(params["sources"] ~= nil and params["sources"] ~= '') then
        parameterText = parameterText .. "-s " .. params["sources"] .. " "
    end
    if(params["lang"] ~= nil and params["lang"] ~= '') then
        parameterText = parameterText .. "-l " .. params["lang"] .. " "
    end
    if(params["ambiguousUninomials"] ~= nil) then
        parameterText = parameterText .. "-A "
    end
    if(params["bytesOffset"] ~= nil) then
        parameterText = parameterText .. "-b "
    end
    if(params["noBayes"] ~= nil) then
        parameterText = parameterText .. "-n "
    end
    if(params["verifierUrl"] ~= nil) then
        parameterText = parameterText .. "-e "
    end
    if(params["verify"] ~= nil) then
        parameterText = parameterText .. "-v "
    end

    outputStream:write(json.encode({args = parameterText}))
    serial:serialize(inputCas:getCas(),outputStream)
end

function deserialize(inputCas, inputStream)
  inputCas:reset()
  deserial:deserialize(inputStream,inputCas:getCas(),true)
end

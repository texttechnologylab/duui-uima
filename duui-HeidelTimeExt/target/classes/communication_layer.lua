-- DUUI communication layer for HeidelTimeExt.
-- This component uses the classic serialize/deserialize mode and transfers the CAS as XMI.

serial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasSerializer")
deserial = luajava.bindClass("org.apache.uima.cas.impl.XmiCasDeserializer")

SUPPORTS_PROCESS = false
SUPPORTS_SERIALIZE = true

function serialize(inputCas, outputStream, params)
    serial:serialize(inputCas:getCas(), outputStream)
end

function deserialize(inputCas, inputStream)
    inputCas:reset()
    deserial:deserialize(inputStream, inputCas:getCas(), true)
end

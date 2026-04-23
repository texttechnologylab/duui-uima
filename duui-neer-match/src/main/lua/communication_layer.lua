-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

function serialize(inputCas, outputStream, parameters)
end

function deserialize(inputCas, inputStream)
end

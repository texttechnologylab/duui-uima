-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
utils = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    local topK = -1
    if params["top_k"] then
        topK = tonumber(params["top_k"])
    end


    local annotation_class_path, success, annotationClass

    if params["annotationClassPath"] then
        annotation_class_path = params["annotationClassPath"]
        success, annotationClass = pcall(luajava.bindClass, annotation_class_path)
    else
        annotation_class_path = ""
        success = false
    end

    local partOfSpeechArray = {}

    if success then
        local partOfSpeechAnnotations = utils:select(inputCas, annotationClass):iterator()
        while partOfSpeechAnnotations:hasNext() do
            local annotation = partOfSpeechAnnotations:next()
            local partOfSpeech = {
                text = annotation:getCoveredText(),
                iBegin = annotation:getBegin(),
                iEnd = annotation:getEnd()
            }
            table.insert(partOfSpeechArray, partOfSpeech)
        end
    else
        print("Cant find annotation" .. annotation_class_path .. ", annotating languages for whole text!")
        local text = inputCas:getDocumentText()
        local partOfSpeech = {
            text=text,
            iBegin=0,
            iEnd=string.len(text)
        }
        table.insert(partOfSpeechArray, partOfSpeech)
    end

    -- Encode data as JSON object and write to stream
    -- TODO Note: The JSON library is automatically included and available in all Lua scripts
    outputStream:write(json.encode({
        part_of_speeches = partOfSpeechArray,
        top_k = topK
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
    for i, language in ipairs(results["languages"]) do
        local languageJavaObject = luajava.newInstance("org.texttechnologylab.annotation.Language", inputCas)
        languageJavaObject:setValue(language["language"])
        languageJavaObject:setScore(language["score"])
        languageJavaObject:setBegin(language["iBegin"])
        languageJavaObject:setEnd(language["iEnd"])
        languageJavaObject:addToIndexes()
    end
end

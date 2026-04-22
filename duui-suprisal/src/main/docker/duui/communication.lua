-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
ConditionSentence = luajava.bindClass("org.texttechnologylab.annotation.neglab.ConditionSentence")
TokenSuprisal = luajava.bindClass("org.texttechnologylab.annotation.neglab.TokenSuprisal")
DocumentMetaData = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")

-- This "serialize" function is called to transform the CAS object into an stream that is sent to the annotator
-- Inputs:
--  - inputCas: The actual CAS object to serialize
--  - outputStream: Stream that is sent to the annotator, can be e.g. a string, JSON payload, ...
function serialize(inputCas, outputStream, params)
    -- Get data from CAS

    local selection_array = {}

    local model = params["model"]
    local token = params["token_authentication"]

    local selectionSet = util:select(inputCas, ConditionSentence):iterator()

    while selectionSet:hasNext() do
        local s = selectionSet:next()

        local tSelection = {
            sText = s:getCoveredText(),
            iBegin = s:getBegin(),
            iEnd = s:getEnd(),
            sCondition = s:getCondition(),
            sTarget = s:getTarget()
        }
        table.insert(selection_array, tSelection)
    end

    -- Encode data as JSON object and write to stream
    outputStream:write(json.encode({
        selection = selection_array,
        model_name = model,
        token = token or ""
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

    if results["error"] then

            for i, error in ipairs(results["error"]) do

                if i>0 then
                    print("===========================================================================")
                elseif i==0 then
                    print("\nMessages")
                end

                print(error["type"] .. ":\t" .. error["error"])
                if(error[cause]) then print(error["cause"]) end

            end
    end

    if results["result"] then

        local model_name = results["result"]["model_name"]

            local dmd = util:selectSingle(inputCas, DocumentMetaData)

            local comment = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            comment:setReference(dmd)
            comment:setKey("model")
            comment:setValue(model_name)
            comment:addToIndexes()

        local r = results["result"]
        for i, token in ipairs(r["tokens"]) do
            local pToken = luajava.newInstance("org.texttechnologylab.annotation.neglab.TokenSuprisal", inputCas)
            pToken:setBegin(token["iBegin"])
            pToken:setEnd(token["iEnd"])
            pToken:setValue(token["sSuprise"])
            pToken:addToIndexes()
        end

        for i, suprise in ipairs(r["sentences"]) do

            local sentence = util:selectSingleAt(inputCas, ConditionSentence, suprise["iBegin"], suprise["iEnd"])

            if sentence ~= nil then
                sentence:setValue(suprise["sSuprise"])
                sentence:setSequenceScore(suprise["mScore"])
                sentence:setSequenceScoreSum(suprise["mScoreSum"])
                sentence:addToIndexes()

            end
        end
    else
print("\n\n")
print("=============== Results ===============")
print([[⡴⠒⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⠉⠳⡆
⣇⠰⠉⢙⡄⠀⠀⣴⠖⢦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣆⠁⠙⡆
⠘⡇⢠⠞⠉⠙⣾⠃⢀⡼⠀⠀⠀⠀⠀⠀⠀⢀⣼⡀⠄⢷⣄⣀⠀⠀⠀⠀⠀⠀⠀⠰⠒⠲⡄⠀⣏⣆⣀⡍
⠀⢠⡏⠀⡤⠒⠃⠀⡜⠀⠀⠀⠀⠀⢀⣴⠾⠛⡁⠀⠀⢀⣈⡉⠙⠳⣤⡀⠀⠀⠀⠘⣆⠀⣇⡼⢋⠀⠀⢱
⠀⠘⣇⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⡴⢋⡣⠊⡩⠋⠀⠀⠀⠣⡉⠲⣄⠀⠙⢆⠀⠀⠀⣸⠀⢉⠀⢀⠿⠀⢸
⠀⠀⠸⡄⠀⠈⢳⣄⡇⠀⠀⢀⡞⠀⠈⠀⢀⣴⣾⣿⣿⣿⣿⣦⡀⠀⠀⠀⠈⢧⠀⠀⢳⣰⠁⠀⠀⠀⣠⠃
⠀⠀⠀⠘⢄⣀⣸⠃⠀⠀⠀⡸⠀⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠈⣇⠀⠀⠙⢄⣀⠤⠚⠁⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⠀⠀⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⢘⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢰⣿⣿⣿⡿⠛⠁⠀⠉⠛⢿⣿⣿⣿⣧⠀⠀⣼⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡀⣸⣿⣿⠟⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⡀⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡇⠹⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⡿⠁⡏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣤⣞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢢⣀⣠⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠲⢤⣀⣀⠀⢀⣀⣀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
]])
    end

end

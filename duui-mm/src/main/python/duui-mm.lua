StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
    --print("start Lua serialzation")
    local doc_lang = inputCas:getDocumentLanguage()

    --print("doc_lang: ", doc_lang)

    -- get the parameters promopt or default
    local prompt = parameters["prompt"] if parameters["prompt"] == nil then prompt = "<grounding>An image of" end
    local model_name = parameters["model_name"] if parameters["model_name"] == nil then model_name = "microsoft/kosmos-2-patch14-224" end
    local selection_types = parameters["selections"] if parameters["selections"] == nil then selection_types="org.texttechnologylab.annotation.type.Image" end
    local individual = parameters["individual"] if parameters["individual"] == nil then individual = "true" end
    local mode = parameters['mode'] if parameters['mode'] == nil then mode = 'simple'

    end
    --print("truncate_text: ", truncate_text)
    --print("start")
    local images = {}
    local number_of_images = 1
    for selection_type in string.gmatch(selection_types, "([^,]+)") do
        print("selection_type: ", selection_type)
        if selection_type == 'text' then
            local doc_text = inputCas:getDocumentText()
            local doc_len = TopicUtils:getDocumentTextLength(inputCas)
            images[number_of_images] = {
                src = doc_text,
                height = 0,
                width = 0,
                begin = 0,
                ['end'] = doc_len
            }
            number_of_images = number_of_images + 1
        else
            local class = Class:forName(selection_type);
            local image_it = JCasUtil:select(inputCas, class):iterator()
            while image_it:hasNext() do
                local image = image_it:next()
                images[number_of_images] = {
                    src = image:getSrc(),
                    height = image:getHeight(),
                    width = image:getWidth(),
                    begin = image:getBegin(),
                    ['end'] = image:getEnd()
                }
                number_of_images = number_of_images + 1
            end

        end

    end

    outputStream:write(json.encode({
        images = images,
        number_of_images = number_of_images,
        prompt = prompt,
        doc_lang = doc_lang,
        model_name = model_name,
        individual = individual,
        mode = mode
    }))
end

function deserialize(inputCas, inputStream)
    print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    --print("results")
    --print(results)

    if results['prompt'] ~= nil then
        local prompt = results['prompt']
        local prompt_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
        prompt_i:setKey("prompt")
        prompt_i:setValue(prompt)
        prompt_i:addToIndexes()
    end

    if results['errors'] ~= nil then
        local errors = results['errors']
        for index_i, error in ipairs(errors) do
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning_i:setKey("error")
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end
    --print("---------------------- Finished errors ----------------------")

    if results['model_source'] ~= nil and results['model_version'] ~= nil and results['model_name'] ~= nil and results['model_lang'] ~= nil then
        --print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

        --print("setMetaData")
        local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
        model_meta:setModelVersion(model_version)
        --         print(model_version)
        model_meta:setModelName(model_name)
        --         print(model_name)
        model_meta:setSource(source)
        --         print(source)
        model_meta:setLang(model_lang)
        --         print(model_lang)
        model_meta:addToIndexes()



        local number_of_images = results["number_of_images"]
        --print("number_of_images: ", number_of_images)


        local results_images = results["images"]
        local results_processed_text = results["processed_text"]
        local results_entities = results["entities"]

        -- remove the prompt subtext from the processed text
        if results['prompt'] ~= nil then
            local prompt = results['prompt']
            results_processed_text = results_processed_text:gsub(prompt, "")
        end

        -- update the document text
        --print("results_processed_text: ", results_processed_text)
        inputCas:setDocumentText(results_processed_text, "plain/text")
        print(inputCas:getDocumentText())

        -- iterate over the results and add them to the CAS
        for index_i, image in ipairs(results_images) do
            -- create image cas object
            local image_i = luajava.newInstance("org.texttechnologylab.annotation.type.Image", inputCas)
            image_i:setSrc(image["src"])
            image_i:setHeight(image["height"])
            image_i:setWidth(image["width"])
            image_i:setBegin(image["begin"])
            image_i:setEnd(image["end"])
            image_i:addToIndexes()

            -- create a subimage using entities
            for entity_name, entity_data in pairs(results_entities) do

                --print("entity_name: ", entity_name)
                local subimage_i = luajava.newInstance("org.texttechnologylab.annotation.type.SubImage", inputCas)
                --print("entity_name: ", entity_name, "type: ", type(entity_name))
                subimage_i:setBegin(entity_data["begin"])
                subimage_i:setEnd(entity_data["end"])
                subimage_i:setParent(image_i)
                --print(entity_data["begin"])
                --print(entity_data["end"])
                --print(entity_data["bounding_box"])
                --print("starting bboxes")
                print("len of bounding_box: ", #entity_data["bounding_box"])
                local coordinates = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #entity_data["bounding_box"])
                if #entity_data["bounding_box"] > 0 then
                    subimage_i:setCoordinates(coordinates)
                    local idx = 0
                    for bx1, bx2 in pairs(entity_data["bounding_box"]) do
                        --print(("idx: %d"):format(idx))
                        --print("x1: ", bx2[1])
                        --print("y1:", bx2[2])
                        --
                        --print("x2: ", bx2[3])
                        --print("y2: ", bx2[4])
                        if idx < #entity_data["bounding_box"] then
                            local coordinate_i = luajava.newInstance("org.texttechnologylab.annotation.type.Coordinate", inputCas)
                            coordinate_i:setX(bx2[1])
                            coordinate_i:setY(bx2[2])
                            coordinate_i:addToIndexes()
                            subimage_i:setCoordinates(idx, coordinate_i)

                        end

                        if (idx + 1) < #entity_data["bounding_box"] then
                            local coordinate_i1 = luajava.newInstance("org.texttechnologylab.annotation.type.Coordinate", inputCas)
                            coordinate_i1:setX(bx2[3])
                            coordinate_i1:setY(bx2[4])
                            coordinate_i1:addToIndexes()
                            subimage_i:setCoordinates(idx + 1, coordinate_i1)
                            idx = idx + 2

                        end
                    end

                end
                --subimage_i:setCoordinates(coordinates)
                subimage_i:addToIndexes()
            end
        end

    end

    -- copy cas into a new one with new document text
end

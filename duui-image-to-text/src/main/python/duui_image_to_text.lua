StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
     print("start Lua serialzation")
    local doc_lang = inputCas:getDocumentLanguage()

    print("doc_lang: ", doc_lang)

    -- get the parameters promopt or default
    local prompt = parameters["prompt"] if parameters["prompt"] == nil then prompt = "<grounding>An image of" end
    local model_name = parameters["model_name"] if parameters["model_name"] == nil then model_name = "microsoft/kosmos-2-patch14-224" end
    --print("truncate_text: ", truncate_text)
    print("start")

    -- loop over the images and get the image annotations
    local image_it = JCasUtil:select(inputCas, luajava.bindClass("org.texttechnologylab.annotation.type.Image")):iterator()
    local number_of_images = 0
    local images_out = {}
    print("hello")
    while image_it:hasNext() do
        print("image_it")
        print("number_of_images: ", number_of_images)
        local image = image_it:next()
        number_of_images = number_of_images + 1
        images_out[number_of_images] = {
            src = image:getSrc(),
            height = image:getHeight(),
            width = image:getWidth(),
            begin = image:getBegin(),
            ['end']= image:getEnd()
        }
    end

    outputStream:write(json.encode({
        images = images_out,
        number_of_images = number_of_images,
        prompt = prompt,
        doc_lang = doc_lang,
        model_name = model_name,
    }))
end

function deserialize(inputCas, inputStream)
    print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    print("results")
    print(results)

    if results['errors'] ~= nil then
        local errors = results['errors']
        for index_i, error in ipairs(errors) do
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning_i:setKey("error")
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end
    print("---------------------- Finished errors ----------------------")

    if results['model_source'] ~= nil and results['model_version'] ~= nil and results['model_name'] ~= nil and results['model_lang'] ~= nil then
         print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

         print("setMetaData")
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
        print("number_of_images: ", number_of_images)

        local results_images = results["images"]
        local results_processed_text = results["processed_text"]
        local results_entities = results["entities"]

        -- update the document text
        print("results_processed_text: ", results_processed_text)
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
                    print("entity_name: ", entity_name)
                    local subimage_i = luajava.newInstance("org.texttechnologylab.annotation.type.SubImage", inputCas)
                    print("entity_name: ", entity_name, "type: ", type(entity_name))
                    subimage_i:setBegin(entity_data["begin"])
                    subimage_i:setEnd(entity_data["end"])
                    subimage_i:setParent(image_i)
                    print(entity_data["begin"])
                    print(entity_data["end"])
                    print(entity_data["bounding_box"])
                    print("starting bboxes")
                    local coordinates = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, 2)
                    subimage_i:setCoordinates(coordinates)
                    print("bbox1, bbox2: ", entity_data["bounding_box"][1], entity_data["bounding_box"][2])
                    local coordinate_i = luajava.newInstance("org.texttechnologylab.annotation.type.Coordinate", inputCas)
                    coordinate_i:setX(entity_data["bounding_box"][1])
                    coordinate_i:setY(entity_data["bounding_box"][2])
                    coordinate_i:addToIndexes()
                    subimage_i:setCoordinates(0, coordinate_i)
                    local coordinate_j = luajava.newInstance("org.texttechnologylab.annotation.type.Coordinate", inputCas)
                    coordinate_j:setX(entity_data["bounding_box"][3])
                    coordinate_j:setY(entity_data["bounding_box"][4])
                    coordinate_j:addToIndexes()
                    subimage_i:setCoordinates(1, coordinate_j)
                    --subimage_i:setCoordinates(coordinates)
                    subimage_i:addToIndexes()
                end
            end

        end

    -- copy cas into a new one with new document text
    end

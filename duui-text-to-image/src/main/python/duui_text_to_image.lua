StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
    local model_name = parameters["model_name"]
    local selection_types = parameters["selection"]
    local image_width = parameters["image_width"] if parameters["image_width"] == nil then image_width = 256 end
    local image_height = parameters["image_height"] if parameters["image_height"] == nil then image_height = 256 end
    local num_inference_steps = parameters["num_inference_steps"] if  parameters["num_inference_steps"] == nil then num_inference_steps = 50 end
    local number_of_images = parameters["number_of_images"] if parameters["number_of_images"] == nil then number_of_images = 1 end
    local low_cpu_mem_usage = parameters["low_cpu_mem_usage"] if parameters["low_cpu_mem_usage"] == nil then low_cpu_mem_usage = false end
    local truncate_text = parameters["truncate_text"] if parameters["truncate_text"] == nil then truncate_text = True end
    --print("truncate_text: ", truncate_text)
    print("start")

    local selections = {}
    local selections_count = 1
    for selection_type in string.gmatch(selection_types, "([^,]+)") do
       local sentences = {}
       if selection_type == "text" then
           local s = {
               text = doc_text,
               begin = 0,
               ['end'] = doc_len
           }
           sentences[1] = s
       else
           --print("start")
           local sentences_count = 1
           local clazz = Class:forName(selection_type);
           local sentences_it = JCasUtil:select(inputCas, clazz):iterator()
           while sentences_it:hasNext() do
               local sentence = sentences_it:next()
               local s = {
                   text = sentence:getCoveredText(),
                   begin = sentence:getBegin(),
                   ['end'] = sentence:getEnd()
               }
               --print(sentence:getCoveredText())
               sentences[sentences_count] = s
               sentences_count = sentences_count + 1
           end
       end

       local selection = {
           sentences = sentences,
           selection = selection_type
       }
       selections[selections_count] = selection
       selections_count = selections_count + 1
    end

    outputStream:write(json.encode({
        selections = selections,
        lang = doc_lang,
        doc_len = doc_len,
        model_name = model_name,
        image_width = image_width,
        image_height = image_height,
        num_inference_steps = num_inference_steps,
        number_of_images = number_of_images,
        low_cpu_mem_usage = low_cpu_mem_usage,
        truncate_text = truncate_text,
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
    if results["results"] ~= nil then
--         print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]

--         print("setMetaData")
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


        local begin_img = results["begin_img"]
--         print("begin_img")
        local end_img = results["end_img"]
--         print("end_img")
        local res_out = results["results"]
--         print("results")
        local res_len = results["len_results"]
--         print("Len_results")
        local factors = results["factors"]

        local number_of_images = results["config"]["number_of_images"]
--      print("number_of_images")
        print("number_of_images: ", number_of_images)

        -- Iterate through res_out with index
        for index_i, res in ipairs(res_out) do
            -- Calculate the index for begin_img and end_img (use the same values for every 2 images)
            local group_index = math.ceil(index_i / number_of_images)
            print("group_index: ", group_index)
            local begin_img_i = begin_img[group_index] -- Use the same begin_img for each group of 2
            local end_img_i = end_img[group_index] -- Use the same end_img for each group of 2


            -- Print debug info
             print("Starting res_out loop")
            -- print(res["src"])
             print("Processing image index", index_i)

            -- Create a new Image annotation object
            local image_i = luajava.newInstance("org.texttechnologylab.annotation.type.Image", inputCas, begin_img_i, end_img_i)
            image_i:setSrc(res["src"])
            image_i:setHeight(res["height"])
            image_i:setWidth(res["width"])

            -- Add the image to indexes
            image_i:addToIndexes()

            -- Optional: Print info for debugging
            -- print("Added image with src", res["src"], "at position", begin_img_i, "to", end_img_i)
        end


        -- model configuration as annotationComment

        -- loop over results[model_config] key value pairs and set them as key value pairs
        for key, value in pairs(results["config"]) do
            local model_config = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas, 0, 0)
            model_config:setKey(key)
            model_config:setValue(value)
            model_config:addToIndexes()
        end

    end
--     print("end")
end

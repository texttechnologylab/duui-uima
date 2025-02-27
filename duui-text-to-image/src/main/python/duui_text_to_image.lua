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
    print("number of images: ", number_of_images)
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
    }))
end

function deserialize(inputCas, inputStream)
    --print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    --print("results")
    --print(results)
    if results["results"] ~= nil then
--         print("GetInfo")
        local source = results["model_source"]
        local model_version = results["model_version"]
        local model_name = results["model_name"]
        local model_lang = results["model_lang"]
        local errors = results["errors"]
        print("errors")
        print(errors)

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

--         print(factors)
        for index_i, res in ipairs(res_out) do
            --print("starting res_out loop")
            --print(res["src"])
            --print("starting .............")
            local begin_img_i = begin_img[index_i]
            --print(begin_img_i)
            local end_img_i = end_img[index_i]
            --print(end_img_i)
            local len_i = res_len[index_i]
            local image_i = luajava.newInstance("org.texttechnologylab.annotation.type.Image", inputCas, begin_img_i, end_img_i)
            image_i:setSrc(res["src"])
            image_i:setHeight(res["height"])
            image_i:setWidth(res["width"])
            image_i:addToIndexes()
        end

        for index_i, error in ipairs(errors) do
            local begin_warning_i = begin_img[index_i]
            local end_warning_i = end_img[index_i]
            print("error")
            print(error)
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas, begin_warning_i, end_warning_i)
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end
--     print("end")
end

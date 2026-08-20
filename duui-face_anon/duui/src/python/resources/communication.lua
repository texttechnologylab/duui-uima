StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Image = luajava.bindClass("org.texttechnologylab.annotation.type.Image")

function serialize(inputCas, outputStream, parameters)
    print("start Lua serialzation")
    -- get the parameters or run default
    local hf_token = parameters["hf_token"]if parameters["hf_token"]==nil then hf_token="None" end
    local anon_type = parameters["anon_type"] if parameters["anon_type"]==nil then anon_type = "single_align" end

    -- blur/pixelate/black only work if the selected method it redaction
    -- redaction can be black, pixelate or blur
    
    local redact_type =  parameters["redact_type"] if parameters["redact_type"]==nil then redact_type = "blur" end
    local face_type = parameters["face_type"] if parameters["face_type"]==nil then face_type = "full_face" end
    local blur = parameters["blur"] if parameters["blur"]==nil then blur=51 end

    local pixel = parameters["pixel"] if parameters["pixel"]==nil then pixel=16 end


    -- all the other possible settings
    local diffusion_model = parameters["diffusion_model"] if parameters["diffusion_model"]==nil then diffusion_model = "sd2-community/stable-diffusion-2-1"  end
    local clip_model = parameters["clip_model"] if parameters["clip_model"]==nil then clip_model = "openai/clip-vit-large-patch14" end
    local seed = parameters["seed"] if parameters["seed"]==nil then seed = 1 end
    local guidance = parameters["guidance"] if parameters["guidance"]==nil then guidance = 4.0 end
    local inference_steps = parameters["inference_steps"] if parameters["inference_steps"]==nil then inference_steps = 25 end
    local anon_degree = parameters["anon_degree"] if parameters["anon_degree"]==nil then anon_degree = 1.25 end
    -- swap has different default settings
    if anon_type == "swap" then
        inference_steps = 200
        anon_degree = 0.0

    end
    local vis_input = parameters["vis_input"] if parameters["vis_input"]==nil then vis_input = "False" end
    local height = parameters["height"] if parameters["height"]==nil then height = tonumber(height) end
    local width = parameters["width"] if parameters["width"]==nil then width = tonumber(height) end
    local frame_interval = parameters["frame_interval"] if parameters["frame_interval"]==nil then frame_interval = 1 end


    --print(anon_type..redact_type..blur..pixel..diffusion_model..clip_model..seed..guidance..inference_steps..anon_degree..vis_input..height..width)
     


    -- input images

    local images = {}
    local number_of_images = 1
    local image_it = JCasUtil:select(inputCas, Image):iterator()
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
    print(images)

    -- input videos
    local videos = {}
    local number_of_videos = 1
    local video_class = Class:forName("org.texttechnologylab.annotation.type.Video")
    local video_it = JCasUtil:select(inputCas, video_class):iterator()
    while video_it:hasNext() do
        local video = video_it:next()
        videos[number_of_videos] = {
            src = video:getSrc(),
            length = video:getLength(),
            fps = video:getFps(),
            begin = video:getBegin(),
            ['end'] = video:getEnd()
        }
        number_of_videos = number_of_videos + 1
    end

    outputStream:write(json.encode({
        anon_type = anon_type,
        anon_degree = anon_degree,
        images = images,
        videos = videos,
        frame_interval = frame_interval,
        redact_type = redact_type,
        face_type = face_type,
        blur = blur,
        pixel = pixel,
        diffusion_model = diffusion_model,
        clip_model = clip_model,
        seed = seed,
        guidance = guidance,
        inference_steps =inference_steps,
        vis_input = vis_input,
        height =height,
        width = width,
        hf_token=hf_token
    }))
    
end

function deserialize(inputCas, inputStream)
    print("start deserialize")
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)
    --print("results")
    --print(results)

    if results['out_errors'] ~= nil then
        local errors = results['out_errors']
        for index_i, error in ipairs(errors) do
            local warning_i = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", inputCas)
            warning_i:setKey("error")
            warning_i:setValue(error)
            warning_i:addToIndexes()
        end
    end
    -- metadata
     --if results['model_source'] ~= nil and results['model_version'] ~= nil and results['model_name'] ~= nil and results['model_lang'] ~= nil then
     --   --print("GetInfo")
     --   local source = results["model_source"]
     --   local model_version = results["model_version"]
     --   local model_name = results["model_name"]
     --   local model_lang = results["model_lang"]
     --
     --   --print("setMetaData")
     --   local model_meta = luajava.newInstance("org.texttechnologylab.annotation.model.MetaData", inputCas)
     --   model_meta:setModelVersion(model_version)
     --   --         print(model_version)
     --   model_meta:setModelName(model_name)
     --   --         print(model_name)
     --   model_meta:setSource(source)
     --   --         print(source)
     --   model_meta:setLang(model_lang)
     --   --         print(model_lang)
     --   model_meta:addToIndexes()

    --end
    -- anonymiyed images
    if results['output_images'] ~= nil then
        local output_images = results['output_images']
        for image_id, img_data in pairs(output_images) do
            local image = luajava.newInstance("org.texttechnologylab.annotation.type.Image", inputCas)
            image:setSrc(img_data["src"])
            image:setWidth(img_data['width'])
            image:setHeight(img_data['height'])
            image:setBegin(img_data['begin'])
            image:setEnd(img_data['end'])
            image:addToIndexes()
        end
    end

    -- anonymized videos
    if results['output_videos'] ~= nil then
        local output_videos = results['output_videos']
        for video_id, video_data in pairs(output_videos) do
            local video = luajava.newInstance("org.texttechnologylab.annotation.type.Video", inputCas)
            video:setSrc(video_data['src'])
            video:setLength(video_data['length'])
            video:setFps(video_data['fps'])
            video:setBegin(video_data['begin'])
            video:setEnd(video_data['end'])
            video:addToIndexes()
        end
    end

end
    --print("---------------------- Finished errors ----------------------")
    

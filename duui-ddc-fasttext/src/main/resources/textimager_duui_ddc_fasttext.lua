XmiCasDeserializer = luajava.bindClass("org.apache.uima.cas.impl.XmiCasDeserializer")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
AnnotationComment = luajava.bindClass("org.texttechnologylab.annotation.AnnotationComment")
OutputKeys = luajava.bindClass("javax.xml.transform.OutputKeys")
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")

DUUIConfiguratinParameterName = "__textimager_duui_configuration_parameter_ddc_fasttext__"
BORLAND_SEP = "¤"

function add_parameter_to_cas(cas, parameters, param_name)
    local param_value = parameters[param_name]
    if param_value ~= nil then
        -- TODO use "AnnotationComment" var somehow?
        local param_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotationComment", cas)
        param_anno:setKey(DUUIConfiguratinParameterName)
        param_anno:setValue(param_name .. BORLAND_SEP .. param_value)
        param_anno:addToIndexes()
        --print("added parameter: " .. param_name)
    end
end

function serialize(inputCas, outputStream, parameters)
    -- Comma separated list of Language and Location and Num Labels of the model.
    -- EX: de,model_de.bin,100,en,en_model1.bin,93...
    --add_parameter_to_cas(inputCas, parameters, "language_models_labels")

    -- Location from which the model is read.
    --add_parameter_to_cas(inputCas, parameters, "fasttext_location")

    -- fastText k parameter == Max Anzahl an Labeln.
    --add_parameter_to_cas(inputCas, parameters, "fasttext_k")

    -- Cutoff all labels with lowest score
    --add_parameter_to_cas(inputCas, parameters, "cutoff")

    -- Comma separated list of Tags to add
    --add_parameter_to_cas(inputCas, parameters, "tags")

    -- Use Lemma instead of Token
    --add_parameter_to_cas(inputCas, parameters, "use_lemma")

    -- Add POS Info to words
    --add_parameter_to_cas(inputCas, parameters, "add_pos")

    -- Location for POS Mapping file
    --add_parameter_to_cas(inputCas, parameters, "posmap_location")

    -- Remove Punctuation from text
    --add_parameter_to_cas(inputCas, parameters, "remove_punct")

    -- Remove Functionwords from text
    --add_parameter_to_cas(inputCas, parameters, "remove_functionwords")

    -- Lazy Load Models
    --add_parameter_to_cas(inputCas, parameters, "lazy_load")

    -- Ignore missing Lemma/POS
    --add_parameter_to_cas(inputCas, parameters, "ignore_missing_lemma_pos")

    -- Max loaded Models
    --add_parameter_to_cas(inputCas, parameters, "lazy_load_max")

    -- Comma separated list of selection to process in order: text (default), paragraph, sentence, token, line (== Div)
    -- Add the div type for "line" with a ";", ex: "sentence,line;newline,token"
    add_parameter_to_cas(inputCas, parameters, "selection")

    -- DDC 2 or DDC 3
    add_parameter_to_cas(inputCas, parameters, "ddc_variant")

    -- add parameters to cas and send cas directly to service
    --XmiCasSerializer:serialize(inputCas:getCas(), outputStream)
    local xmlSerializer = luajava.newInstance("org.apache.uima.util.XMLSerializer", outputStream, true)
    xmlSerializer:setOutputProperty(OutputKeys.VERSION, "1.1")
    xmlSerializer:setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8)
    local xmiCasSerializer = luajava.newInstance("org.apache.uima.cas.impl.XmiCasSerializer", nil)
    xmiCasSerializer:serialize(inputCas:getCas(), xmlSerializer:getContentHandler())
end

function deserialize(inputCas, inputStream)
    -- returns full cas directly and completely overwrite existing cas
    inputCas:reset()
    XmiCasDeserializer:deserialize(inputStream, inputCas:getCas(), true)

    -- remove all confguration parameters
    local params_to_remove = {}
    local param_anno_it = JCasUtil:select(inputCas, AnnotationComment):iterator()
    while param_anno_it:hasNext() do
        local param_anno = param_anno_it:next()
        if param_anno:getKey() == DUUIConfiguratinParameterName then
            table.insert(params_to_remove, param_anno)
        end
    end
    for _, param_anno in ipairs(params_to_remove)
    do
        param_anno:removeFromIndexes()
    end
end

-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
AnnotationComment = luajava.bindClass("org.texttechnologylab.annotation.AnnotationComment")
Taxon = luajava.bindClass("org.texttechnologylab.annotation.type.Taxon")
Object = luajava.bindClass("java.lang.Object")

function selectAnnotationComments(view)
    local selection_iterator = JCasUtil:select(view, AnnotationComment:class)
    local annotation_comments = {}
    while selection_iterator:hasNext() do
        local annotation_comment = selection_iterator:next()
        local ref = annotation_comment:getReference()
        if (Taxon:class:isInstance(ref)) then
            table.insert(annotation_comments, annotation_comment)
        end
    end
    return annotation_comments
end

function serialize(inputCas, outputStream, parameters)
    local document_text = inputCas:getDocumentText()
    local annotations_view_name = parameters["annotations_view"]
    local annotations_view = inputCas
    if annotations_view_name ~= nil and annotations_view_name ~= "" then
        annotations_view = inputCas:getView(annotations_view_name)
    end
    local annotation_comments = selectAnnotationComments(annotations_view)
    local recognized_taxa = {}
    for _, annotation_comment in ipairs(annotation_comments) do
        local taxon = Taxon:class:cast(annotation_comment:getReference())
        local begin = taxon:getBegin()
        -- insert taxon collection by begin position, if not already present
        local recognized_taxon = recognized_taxa[begin]
        if recognized_taxon == nil then
            local end = taxon:getEnd()
            local text = document_text:sub(begin + 1, end) -- Lua strings are 1-indexed
            local 
            recognized_taxon = {
                begin = begin,
                end = end,
                text = text,
                linkings = {}
            }
            recognized_taxa[begin] = recognized_taxon
        end
        local comment_key = annotation_comment:getKey()
        if Object:equals(comment_key, "linking") then
            local comment_value = annotation_comment:getValue()
            table.insert(recognized_taxon.linkings, comment_value)
        end
    end
    local recognized_taxa_list = {}
    for _, recognized_taxon in pairs(recognized_taxa) do
        table.insert(recognized_taxa_list, recognized_taxon)
    end

    outputStream:write(json.encode({
        taxa = recognized_taxa_list
    }))
end

function deserialize(inputCas, inputStream)
    local input_bytes = inputStream:readAllBytes()
    local input_string = luajava.newInstance("java.lang.String", input_bytes, StandardCharsets.UTF_8)
    print("Deserialized input: " .. input_string)
end

-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
ClassLoader = luajava.bindClass("java.lang.ClassLoader")
SystemClassLoader = ClassLoader:getSystemClassLoader()
AnnotationComment = luajava.bindClass("org.texttechnologylab.annotation.AnnotationComment")
AnnotationCommentClass = Class:forName("org.texttechnologylab.annotation.AnnotationComment")
Taxon = luajava.bindClass("org.texttechnologylab.annotation.type.Taxon")
TaxonClass = Class:forName("org.texttechnologylab.annotation.type.Taxon")
Object = luajava.bindClass("java.lang.Object")

function instanceOf(clazz, object)
    -- ok this is really bad, but I could not find any other way
    local object_class = object:getClass()
    local object_class_name = tostring(object_class)
    local clazz_name = tostring(clazz)
    local is_instance = object_class_name == clazz_name
    return is_instance
end

function selectAnnotationComments(view)
    local selection_iterator = JCasUtil:select(view, AnnotationCommentClass):iterator()
    local annotation_comments = {}
    while selection_iterator:hasNext() do
        local annotation_comment = selection_iterator:next()
        local ref = annotation_comment:getReference()
        -- TaxonClass:isAssignableFrom(ref_class) does not work
        -- TaxonClass:isInstance(ref) does not work
        -- Object:equals(ref_class, TaxonClass) does not throw, but always returns false, even if exactly the same
        if (instanceOf(TaxonClass, ref)) then
            table.insert(annotation_comments, annotation_comment)
        end
    end
    return annotation_comments
end

function serialize(inputCas, outputStream, parameters)
    local document_text = inputCas:getDocumentText()
    -- print(document_text:sub(1, 300))
    local test_string = "äöüß"
    print(test_string)
    local annotations_view_name = parameters["annotations_view"]
    local annotations_view = inputCas
    if annotations_view_name ~= nil and annotations_view_name ~= "" then
        annotations_view = inputCas:getView(annotations_view_name)
    end
    local annotation_comments = selectAnnotationComments(annotations_view)
    local recognized_taxa = {}
    for _, annotation_comment in ipairs(annotation_comments) do
        -- local taxon = TaxonClass:cast(annotation_comment:getReference())
        local taxon = annotation_comment:getReference()
        local begin = taxon:getBegin()
        -- insert taxon collection by begin position, if not already present
        local recognized_taxon = recognized_taxa[begin]
        if recognized_taxon == nil then
            local end_ = taxon:getEnd()
            -- local text = document_text:sub(begin + 1, end_)
            -- local text = taxon:getCoveredText()
            -- print("Recognized taxon: " .. text .. " (begin: " .. begin .. ", end: " .. end_ .. ")")
            recognized_taxon = {
                -- text = text,
                linkings = {}
            }
            recognized_taxon["begin"] = begin
            recognized_taxon["end"] = end_
            recognized_taxa[begin] = recognized_taxon
        end
        local comment_key = annotation_comment:getKey()
        if comment_key == "linking" then
            local comment_value = annotation_comment:getValue()
            table.insert(recognized_taxon.linkings, comment_value)
        end
    end
    local recognized_taxa_list = {}
    for _, recognized_taxon in pairs(recognized_taxa) do
        table.insert(recognized_taxa_list, recognized_taxon)
    end

    -- print("Recognized taxa: " .. json.encode(recognized_taxa_list))

    outputStream:write(json.encode({
        taxa = recognized_taxa_list,
        document_text = document_text
    }))
end

function deserialize(inputCas, inputStream)
    local input_bytes = inputStream:readAllBytes()
    local input_string = luajava.newInstance("java.lang.String", input_bytes, StandardCharsets.UTF_8)
    print("Deserialized input: " .. input_string)
end

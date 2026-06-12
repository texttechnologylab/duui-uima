-- Bind static classes from java
StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
Class = luajava.bindClass("java.lang.Class")
AnnotationCommentClass = Class:forName("org.texttechnologylab.annotation.AnnotationComment")
TaxonClass = Class:forName("org.texttechnologylab.annotation.type.Taxon")

function instanceOf(clazz, object)
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
        if (instanceOf(TaxonClass, ref)) then
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
        local taxon = annotation_comment:getReference()
        local begin = taxon:getBegin()
        -- insert taxon collection by begin position, if not already present
        local recognized_taxon = recognized_taxa[begin]
        if recognized_taxon == nil then
            local end_ = taxon:getEnd()
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

    outputStream:write(json.encode({
        taxa = recognized_taxa_list,
        document_text = document_text
    }))
end

function populateTaxonResolution(taxon_resolution, properties)
    taxon_resolution:setProvider(properties["provider"])
    taxon_resolution:setTaxonId(properties["taxon_id"])
    
    local kingdom_name = properties["kingdom_name"]
    if kingdom_name ~= nil then
        taxon_resolution:setKingdomName(kingdom_name)
    end
    local kingdom_id = properties["kingdom_key"]
    if kingdom_id ~= nil then
        taxon_resolution:setKingdomId(kingdom_id)
    else
        taxon_resolution:setKingdomId(-1)
    end
    local phylum_name = properties["phylum_name"]
    if phylum_name ~= nil then
        taxon_resolution:setPhylumName(phylum_name)
    end
    local phylum_id = properties["phylum_key"]
    if phylum_id ~= nil then
        taxon_resolution:setPhylumId(phylum_id)
    else
        taxon_resolution:setPhylumId(-1)
    end
    local class_name = properties["class_name"]
    if class_name ~= nil then
        taxon_resolution:setClassName(class_name)
    end
    local class_id = properties["class_key"]
    if class_id ~= nil then
        taxon_resolution:setClassId(class_id)
    else
        taxon_resolution:setClassId(-1)
    end
    local order_name = properties["order_name"]
    if order_name ~= nil then
        taxon_resolution:setOrderName(order_name)
    end
    local order_id = properties["order_key"]
    if order_id ~= nil then
        taxon_resolution:setOrderId(order_id)
    else
        taxon_resolution:setOrderId(-1)
    end
    local superfamily_name = properties["superfamily_name"]
    if superfamily_name ~= nil then
        taxon_resolution:setSuperfamilyName(superfamily_name)
    end
    local superfamily_id = properties["superfamily_key"]
    if superfamily_id ~= nil then
        taxon_resolution:setSuperfamilyId(superfamily_id)
    else
        taxon_resolution:setSuperfamilyId(-1)
    end
    local family_name = properties["family_name"]
    if family_name ~= nil then
        taxon_resolution:setFamilyName(family_name)
    end
    local family_id = properties["family_key"]
    if family_id ~= nil then
        taxon_resolution:setFamilyId(family_id)
    else
        taxon_resolution:setFamilyId(-1)
    end
    local subfamily_name = properties["subfamily_name"]
    if subfamily_name ~= nil then
        taxon_resolution:setSubfamilyName(subfamily_name)
    end
    local subfamily_id = properties["subfamily_key"]
    if subfamily_id ~= nil then
        taxon_resolution:setSubfamilyId(subfamily_id)
    else
        taxon_resolution:setSubfamilyId(-1)
    end
    local tribe_name = properties["tribe_name"]
    if tribe_name ~= nil then
        taxon_resolution:setTribeName(tribe_name)
    end
    local tribe_id = properties["tribe_key"]
    if tribe_id ~= nil then
        taxon_resolution:setTribeId(tribe_id)
    else
        taxon_resolution:setTribeId(-1)
    end
    local subtribe_name = properties["subtribe_name"]
    if subtribe_name ~= nil then
        taxon_resolution:setSubtribeName(subtribe_name)
    end
    local subtribe_id = properties["subtribe_key"]
    if subtribe_id ~= nil then
        taxon_resolution:setSubtribeId(subtribe_id)
    else
        taxon_resolution:setSubtribeId(-1)
    end
    local genus_name = properties["genus_name"]
    if genus_name ~= nil then
        taxon_resolution:setGenusName(genus_name)
    end
    local genus_id = properties["genus_key"]
    if genus_id ~= nil then
        taxon_resolution:setGenusId(genus_id)
    else
        taxon_resolution:setGenusId(-1)
    end
    local subgenus_name = properties["subgenus_name"]
    if subgenus_name ~= nil then
        taxon_resolution:setSubgenusName(subgenus_name)
    end
    local subgenus_id = properties["subgenus_key"]
    if subgenus_id ~= nil then
        taxon_resolution:setSubgenusId(subgenus_id)
    else
        taxon_resolution:setSubgenusId(-1)
    end
    local species_name = properties["species_name"]
    if species_name ~= nil then
        taxon_resolution:setSpeciesName(species_name)
    end
    local species_id = properties["species_key"]
    if species_id ~= nil then
        taxon_resolution:setSpeciesId(species_id)
    else
        taxon_resolution:setSpeciesId(-1)
    end
    local parent_name = properties["parent_name"]
    if parent_name ~= nil then
        taxon_resolution:setParentName(parent_name)
    end
    local parent_id = properties["parent_key"]
    if parent_id ~= nil then
        taxon_resolution:setParentId(parent_id)
    else
        taxon_resolution:setParentId(-1)
    end

    local scientific_name = properties["scientific_name"]
    if scientific_name ~= nil then
        taxon_resolution:setScientificName(scientific_name)
    end
    local canonical_name = properties["canonical_name"]
    if canonical_name ~= nil then
        taxon_resolution:setCanonicalName(canonical_name)
    end
    local vernacular_name = properties["vernacular_name"]
    if vernacular_name ~= nil then
        taxon_resolution:setVernacularName(vernacular_name)
    end
    local accepted_name_usage = properties["accepted_name_usage"]
    if accepted_name_usage ~= nil then
        taxon_resolution:setAcceptedNameUsage(accepted_name_usage)
    end
    local authorship = properties["authorship"]
    if authorship ~= nil then
        taxon_resolution:setAuthorship(authorship)
    end
    local name_type = properties["name_type"]
    if name_type ~= nil then
        taxon_resolution:setNameType(name_type)
    end
    local rank = properties["rank"]
    if rank ~= nil then
        taxon_resolution:setRank(rank)
    end
    local origin = properties["origin"]
    if origin ~= nil then
        taxon_resolution:setOrigin(origin)
    end
    local taxonomic_status = properties["taxonomic_status"]
    if taxonomic_status ~= nil then
        taxon_resolution:setTaxonomicStatus(taxonomic_status)
    end
    local remarks = properties["remarks"]
    if remarks ~= nil then
        taxon_resolution:setRemarks(remarks)
    end
    local references = properties["references"]
    if references ~= nil then
        taxon_resolution:setReferences(references)
    end
    local published_in = properties["published_in"]
    if published_in ~= nil then
        taxon_resolution:setPublishedIn(published_in)
    end
    local num_descendants = properties["num_descendants"]
    if num_descendants ~= nil then
        taxon_resolution:setNumDescendants(num_descendants)
    else
        taxon_resolution:setNumDescendants(-1)
    end
    local last_crawled = properties["last_crawled"]
    if last_crawled ~= nil then
        taxon_resolution:setLastCrawled(last_crawled)
    end
    local last_interpreted = properties["last_interpreted"]
    if last_interpreted ~= nil then
        taxon_resolution:setLastInterpreted(last_interpreted)
    end
    local species_epithet = properties["species_epithet"]
    if species_epithet ~= nil then
        taxon_resolution:setSpeciesEpithet(species_epithet)
    end
    local infraspecific_epithet = properties["infraspecific_epithet"]
    if infraspecific_epithet ~= nil then
        taxon_resolution:setInfraspecificEpithet(infraspecific_epithet)
    end
    local cultivar_epithet = properties["cultivar_epithet"]
    if cultivar_epithet ~= nil then
        taxon_resolution:setCultivarEpithet(cultivar_epithet)
    end
    local url = properties["url"]
    if url ~= nil then
        taxon_resolution:setUrl(url)
    end
    local wikidata_id = properties["wikidata_id"]
    if wikidata_id ~= nil then
        taxon_resolution:setWikidataId(wikidata_id)
    end
    local wikidata_url = properties["wikidata_url"]
    if wikidata_url ~= nil then
        taxon_resolution:setWikidataUrl(wikidata_url)
    end
end

function deserialize(inputCas, inputStream)
    local input_string = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(input_string)

    for _, taxon in ipairs(results["taxa"] or {}) do
        local begin = taxon["begin"]
        local end_ = taxon["end"]
        local recognized_taxon = luajava.newInstance("org.texttechnologylab.annotation.type.RecognizedTaxon", inputCas)
        recognized_taxon:setBegin(begin)
        recognized_taxon:setEnd(end_)
        recognized_taxon:setText(taxon["text"])
        local linkings = taxon["resolved_linkings"]
        local resolutions = luajava.newInstance("org.apache.uima.jcas.cas.FSArray", inputCas, #linkings)
        recognized_taxon:setResolutions(resolutions)
        recognized_taxon:addToIndexes()

        for i, linking in ipairs(linkings) do
            local taxon_resolution = luajava.newInstance("org.texttechnologylab.annotation.type.TaxonResolution", inputCas)
            taxon_resolution:setBegin(begin)
            taxon_resolution:setEnd(end_)
            taxon_resolution:setRecognizedTaxon(recognized_taxon)
            populateTaxonResolution(taxon_resolution, linking)
            taxon_resolution:addToIndexes()
            resolutions:set(i - 1, taxon_resolution)
        end
    end

end

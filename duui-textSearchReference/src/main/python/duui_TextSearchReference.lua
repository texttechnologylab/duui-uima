StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
util = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
TopicUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
TextSearchs = luajava.bindClass("org.texttechnologylab.type.search.TextSearch")

function serialize(inputCas, outputStream, parameters)
--     print("start")
    local doc_lang = inputCas:getDocumentLanguage()
    local doc_text = inputCas:getDocumentText()
    local doc_len = TopicUtils:getDocumentTextLength(inputCas)
--     print(doc_len)
--     print(model_name)
    local method = parameters["method"]
    local search = parameters["search"]
    local search_language = parameters["search_language"]


--     print(model_name)
    local all_searches = {}
--     print(select)

    local selections = {}
    local selections_count = 1
    local search_counter = 0
    local search_in = util:select(inputCas, TextSearchs):iterator()
    while search_in:hasNext() do
        local search = search_in:next()
        local begin_search = search:getBegin()
        local end_search = search:getEnd()
        local search_text = search:getText()
        all_searches[selections_count] = {
            text = search_text,
            begin = begin_search,
            id = search_counter,
            ['end'] = end_search
        }
        search_counter = search_counter + 1
        selections_count = selections_count + 1
    end
--     print(all_prompts)
    outputStream:write(json.encode({
        text = doc_text,
        lang = doc_lang,
        len = doc_len,
        searches = all_searches,
        search_language = search_language,
        method = method,
        search = search
    }))
end

-- This "deserialize" function is called on receiving the results from the annotator that have to be transformed into a CAS object
-- Inputs:
--  - inputCas: The actual CAS object to deserialize into
--  - inputStream: Stream that is received from to the annotator, can be e.g. a string, JSON payload, ...
function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    if results["texts"] ~= nill and results["meta"] ~=nil and results["modification_meta"]~= nil then
--         print("GetInfo")
        local  modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local meta = results["meta"]
        local references_begin=results["references_begin"]
        local references_end=results["references_end"]
        local references_ids=results["references_ids"]
        local urls=results["urls"]
        local groups=results["groups"]
        local methods=results["methods"]
        local priorities=results["priorities"]
        local summaries=results["summaries"]
        local infos=results["infos"]
        local texts=results["texts"]
        local success =results["success"]
        local datetimes=results["datetimes"]

        for index_i, ref_begin in ipairs(references_begin) do
            local begin_i = ref_begin
--             print("begin_prompt_i")
            local end_i = references_end[index_i]
--             print(end_prompt_i)
            local ref_id_i = references_ids
            local url_i = urls[index_i]
            local group_i = groups[index_i]
            local method_i = methods[index_i]
            local priority_i = priorities[index_i]
            local summary_i = summaries[index_i]
            local info_i = infos[index_i]
            local text_i = texts[index_i]
            local success_i = success[index_i]
            local datetime_i = datetimes[index_i]


            local search_text_i = util:selectByIndex(inputCas, TextSearchs, ref_id_i)

            local refText = luajava.newInstance("org.texttechnologylab.annotation.search.ReferenceText", inputCas)
            refText:setBegin(begin_i)
            refText:setEnd(end_i)
            refText:setMethods(method_i)
            refText:setGroup(group_i)
            refText:setText(text_i)
            refText:setUrl(url_i)
            refText:setSuccess(success_i)
            refText:setPriority(priority_i)
            refText:setDateTime(datetime_i)
            refText:setSummary(summary_i)
            refText:setInfos(info_i)
            refText:setReference(search_text_i)
            refText:addToIndexes()
        end
    end
end

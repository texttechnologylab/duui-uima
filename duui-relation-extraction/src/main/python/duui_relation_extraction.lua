-- DUUI Lua mapping for Relation Extraction
-- Maps DUUIRequest → Python
-- Maps Python Response → DUUI Annotations

function map_request(doc)
    return {
        doc_len = #doc.text,
        lang = doc.lang or "en",
        selections = doc.sentences
    }
end

function map_response(response)
    local annotations = {}
    for i, sentence_relations in ipairs(response.relations) do
        for _, triplet in ipairs(sentence_relations) do
            table.insert(annotations, {
                subject = triplet.subject,
                predicate = triplet.predicate,
                object = triplet.object,
                confidence = triplet.confidence
            })
        end
    end
    return annotations
end
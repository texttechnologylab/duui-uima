StandardCharsets = luajava.bindClass("java.nio.charset.StandardCharsets")
Class = luajava.bindClass("java.lang.Class")
Double = luajava.bindClass("java.lang.Double")
JCasUtil = luajava.bindClass("org.apache.uima.fit.util.JCasUtil")
DUUIUtils = luajava.bindClass("org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaUtils")
Token = luajava.bindClass("org.texttechnologylab.uima.type.spacy.SpacyToken")
NounChunk = luajava.bindClass("org.texttechnologylab.uima.type.spacy.SpacyNounChunk")
Sentence = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
Paragraph = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph")
Dependency = luajava.bindClass("de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency")

COHMETRIX_TYPE_PREFIX = "org.texttechnologylab.uima.type.cohmetrix."

-- The concrete UIMA type follows the stable Coh-Metrix 3 label. Project-only
-- indices without a V3 label use their TTLab label instead.
function resolveIndexTypeName(index)
    local label = index["label_v3"]

    if label == nil
        or label == ""
        or string.lower(label) == "n/a"
        or label == "-"
    then
        label = index["label_ttlab"]
    end

    if label == nil or label == "" then
        error("Cannot resolve Coh-Metrix UIMA type: both label_v3 and label_ttlab are missing")
    end

    -- All generated type names are based on labels and must be valid UIMA
    -- short names. Reject malformed labels instead of silently writing the
    -- annotation as the generic Index type.
    if string.match(label, "^[A-Za-z_][A-Za-z0-9_]*$") == nil then
        error("Invalid Coh-Metrix UIMA type label: " .. tostring(label))
    end

    return COHMETRIX_TYPE_PREFIX .. label
end

function requireFeature(index_type, feature_name)
    local feature = index_type:getFeatureByBaseName(feature_name)
    if feature == nil then
        error(
            "Coh-Metrix UIMA type "
            .. index_type:getName()
            .. " does not provide inherited feature "
            .. feature_name
        )
    end
    return feature
end

function setOptionalStringFeature(feature_structure, feature, value)
    if value ~= nil then
        feature_structure:setStringValue(feature, value)
    end
end

function serialize(inputCas, outputStream, parameters)
    local paragraphs = {}
    local paragraphs_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, Paragraph)):listIterator()
    while paragraphs_it:hasNext() do
        local paragraph = paragraphs_it:next()
        local paragraph_data = {
            begin = paragraph:getBegin(),
            ['end'] = paragraph:getEnd(),
            text = paragraph:getCoveredText(),
            sentences = {}
        }
        local sentences_it = luajava.newInstance("java.util.ArrayList", JCasUtil:selectCovered(Sentence, paragraph)):listIterator()
        while sentences_it:hasNext() do
            local sentence = sentences_it:next()
            local sentence_data = {
                begin = sentence:getBegin(),
                ['end'] = sentence:getEnd(),
                text = sentence:getCoveredText(),
                tokens = {}
            }
            local sentence_tokens = luajava.newInstance(
				"java.util.ArrayList",
				JCasUtil:selectCovered(Token, sentence)
			)

			local tokens_it = sentence_tokens:listIterator()
            while tokens_it:hasNext() do
				-- FV3 fix: Store the zero-based sentence-local token index so the
				-- dependency governor can be transferred to Python as head_index.
				local token_index = tokens_it:nextIndex()
				local token = tokens_it:next()

				local dep_type = ""
				local head_index = nil

				local deps_it = luajava.newInstance(
					"java.util.ArrayList",
					JCasUtil:selectCovered(Dependency, sentence)
				):listIterator()

				while deps_it:hasNext() do
					local dep = deps_it:next()

					if dep:getDependent() == token then
						dep_type = dep:getDependencyType()

						local governor = dep:getGovernor()

						if governor ~= nil then
							-- Prefer the actual feature-structure identity. A span alone is
							-- ambiguous when retokenization creates multiple tokens with the
							-- same begin/end offsets.
							for i = 0, sentence_tokens:size() - 1 do
								local candidate = sentence_tokens:get(i)

								if candidate == governor then
									head_index = i
									break
								end
							end

							-- Compatibility fallback for CAS implementations whose Lua
							-- wrappers do not preserve proxy identity. Accept a span match
							-- only when it identifies exactly one sentence token.
							if head_index == nil then
								local matching_index = nil
								local matching_count = 0

								for i = 0, sentence_tokens:size() - 1 do
									local candidate = sentence_tokens:get(i)

									if candidate:getBegin() == governor:getBegin()
										and candidate:getEnd() == governor:getEnd()
									then
										matching_index = i
										matching_count = matching_count + 1
									end
								end

								if matching_count == 1 then
									head_index = matching_index
								end
							end
						end

						-- Fallback for ROOT annotations if no governor could be resolved.
						if head_index == nil
							and (
								dep_type == "--"
								or dep_type == "ROOT"
								or dep_type == "root"
							)
						then
							head_index = token_index
						end

						break
					end
				end

                local vector = nil
                local has_vector = token:getHasVector()
                if has_vector then
                    vector = {}
                    local vector_it = token:getVector():iterator();
                    while vector_it:hasNext() do
                        local vec = vector_it:next()
                        vector[#vector + 1] = vec
                    end
                end

                local token_data = {
                    begin = token:getBegin(),
                    ['end'] = token:getEnd(),
                    text = token:getCoveredText(),
                    lemma = token:getLemmaValue(),
                    pos_value = token:getPos():getPosValue(),
                    pos_coarse = token:getPos():getCoarseValue(),
                    is_alpha = token:getIsAlpha(),
                    is_punct = token:getIsPunct(),
                    dep_type = dep_type,
					head_index = head_index,
                    morph_person = token:getMorph():getPerson(),
                    morph_number = token:getMorph():getNumber(),
                    morph_tense = token:getMorph():getTense(),
                    vector = vector,
                    has_vector = has_vector,
                }
                sentence_data.tokens[#sentence_data.tokens + 1] = token_data
            end
            paragraph_data.sentences[#paragraph_data.sentences + 1] = sentence_data
        end
        paragraphs[#paragraphs + 1] = paragraph_data
    end

    local noun_chunks = {}
    local noun_chunks_it = luajava.newInstance("java.util.ArrayList", JCasUtil:select(inputCas, NounChunk)):listIterator()
    while noun_chunks_it:hasNext() do
        local noun_chunk = noun_chunks_it:next()
        local noun_chunk_data = {
            begin = noun_chunk:getBegin(),
            ['end'] = noun_chunk:getEnd(),
        }
        noun_chunks[#noun_chunks + 1] = noun_chunk_data
    end

    outputStream:write(json.encode({
        text = inputCas:getDocumentText(),
        language = inputCas:getDocumentLanguage(),
        paragraphs = paragraphs,
        noun_chunks = noun_chunks,
    }))
end

function deserialize(inputCas, inputStream)
    local inputString = luajava.newInstance("java.lang.String", inputStream:readAllBytes(), StandardCharsets.UTF_8)
    local results = json.decode(inputString)

    local doc_len = DUUIUtils:getDocumentTextLength(inputCas)

    if results["modification_meta"] ~= nil and results["meta"] ~= nil and results["indices"] ~= nil then
        local meta = results["meta"]

        local modification_meta = results["modification_meta"]
        local modification_anno = luajava.newInstance("org.texttechnologylab.annotation.DocumentModification", inputCas)
        modification_anno:setUser(modification_meta["user"])
        modification_anno:setTimestamp(modification_meta["timestamp"])
        modification_anno:setComment(modification_meta["comment"])
        modification_anno:addToIndexes()

        local cas = inputCas:getCas()
        local type_system = cas:getTypeSystem()

        for i, index in ipairs(results["indices"]) do
            local index_type_name = resolveIndexTypeName(index)
            local index_type = type_system:getType(index_type_name)

            if index_type == nil then
                error(
                    "Coh-Metrix UIMA type is missing from TypeSystem.xml: "
                    .. index_type_name
                )
            end

            -- Create the subtype dynamically through the CAS API. This avoids
            -- requiring one generated Java/JCas class for every Coh-Metrix
            -- index while retaining all features inherited from Index.
            local index_anno = cas:createAnnotation(index_type, 0, doc_len)

            local index_feature = requireFeature(index_type, "index")
            local type_name_feature = requireFeature(index_type, "typeName")
            local label_ttlab_feature = requireFeature(index_type, "labelTTLab")
            local label_v3_feature = requireFeature(index_type, "labelV3")
            local label_v2_feature = requireFeature(index_type, "labelV2")
            local description_feature = requireFeature(index_type, "description")
            local value_feature = requireFeature(index_type, "value")
            local error_feature = requireFeature(index_type, "error")
            local version_feature = requireFeature(index_type, "version")

            index_anno:setIntValue(index_feature, index["index"])
            setOptionalStringFeature(index_anno, type_name_feature, index["type_name"])
            setOptionalStringFeature(index_anno, label_ttlab_feature, index["label_ttlab"])
            setOptionalStringFeature(index_anno, label_v3_feature, index["label_v3"])
            setOptionalStringFeature(index_anno, label_v2_feature, index["label_v2"])
            setOptionalStringFeature(index_anno, description_feature, index["description"])
            -- UIMA's primitive double feature cannot represent JSON null.
            -- Preserve "not computable" as NaN so it cannot silently become
            -- the valid, calculated result 0.0 in the CAS.
            if index["value"] == nil then
                index_anno:setDoubleValue(value_feature, Double.NaN)
            else
                index_anno:setDoubleValue(value_feature, index["value"])
            end
            setOptionalStringFeature(index_anno, error_feature, index["error"])
            setOptionalStringFeature(index_anno, version_feature, index["version"])
            cas:addFsToIndexes(index_anno)

            local meta_anno = luajava.newInstance("org.texttechnologylab.annotation.AnnotatorMetaData", inputCas)
            meta_anno:setReference(index_anno)
            meta_anno:setName(meta["name"])
            meta_anno:setVersion(meta["version"])
            meta_anno:setModelName(meta["modelName"])
            meta_anno:setModelVersion(meta["modelVersion"])
            meta_anno:addToIndexes()
        end
    end
end

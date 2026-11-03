package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.Type;
import org.apache.uima.cas.text.AnnotationFS;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasUtil;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FloatArray;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * spaCy on the words of duui-stanza (parameter {@code token_type}): a SpacyToken and a WordEmbedding per Stanza word, so
 * the two words of a split contraction get their own vectors.
 * <p>Needs both services, URLs from the system properties {@code stanza.url} (default {@code http://localhost:9806})
 * and {@code spacy.url} (default {@code http://localhost:9801}). The Stanza and spaCy types are read through the CAS API.
 */
public class SpaCyStanzaTokensTest {

    private static final String STANZA_URL = System.getProperty("stanza.url", "http://localhost:9806");
    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");
    private static final String STANZA_TOKEN = "org.texttechnologylab.uima.type.stanza.StanzaToken";
    private static final String SPACY_TOKEN = "org.texttechnologylab.uima.type.spacy.SpacyToken";
    private static final String WORD_EMBEDDING = "org.texttechnologylab.uima.type.WordEmbedding";

    private static TypeSystemDescription typeSystem;

    private static DUUIComposer composer(Map<String, String> spacyParameters) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component(STANZA_URL).build().withTimeout(600L));
        DUUIRemoteDriver.Component spacy = new DUUIRemoteDriver.Component(SPACY_URL)
                .withParameter("model_variant", "large")
                .withParameter("use_existing_tokens", "true")
                .withParameter("use_existing_sentences", "true")
                .withParameter("token_type", STANZA_TOKEN);
        spacyParameters.forEach(spacy::withParameter);
        composer.add(spacy.build().withTimeout(600L));
        return composer;
    }

    @BeforeAll
    public static void typeSystem() throws Exception {
        // the CAS needs the type systems of both components (StanzaToken, SpacyToken, WordEmbedding)
        DUUIComposer composer = composer(Map.of());
        try {
            typeSystem = composer.instantiate_pipeline();
        } finally {
            composer.shutdown();
        }
    }

    /** "Er geht zum Haus ." with one plain Sentence and a plain Token per word, as SoMaJo would give them. */
    private static JCas run(Map<String, String> spacyParameters) throws Exception {
        String text = "Er geht zum Haus .";
        JCas cas = JCasFactory.createJCas(typeSystem);
        cas.setDocumentText(text);
        cas.setDocumentLanguage("de");
        new Sentence(cas, 0, text.length()).addToIndexes();
        int begin = 0;
        for (String word : text.split(" ")) {
            new Token(cas, begin, begin + word.length()).addToIndexes();
            begin += word.length() + 1;
        }
        DUUIComposer composer = composer(spacyParameters);
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
        return cas;
    }

    private static List<AnnotationFS> select(JCas cas, String typeName) {
        Type type = cas.getTypeSystem().getType(typeName);
        assertNotNull(type, typeName);
        return new ArrayList<>(CasUtil.select(cas.getCas(), type));
    }

    @Test
    public void wordsOfAContractionGetTheirOwnVectors() throws Exception {
        JCas cas = run(Map.of());

        List<AnnotationFS> stanzaTokens = select(cas, STANZA_TOKEN);
        List<AnnotationFS> spacyTokens = select(cas, SPACY_TOKEN);
        // Er geht zu dem Haus . : six Stanza words, a SpacyToken for each of them
        assertEquals(6, stanzaTokens.size());
        assertEquals(stanzaTokens.size(), spacyTokens.size());
        for (int i = 0; i < stanzaTokens.size(); i++) {
            assertEquals(stanzaTokens.get(i).getBegin(), spacyTokens.get(i).getBegin());
            assertEquals(stanzaTokens.get(i).getEnd(), spacyTokens.get(i).getEnd());
        }

        // the two WordEmbeddings over "zum": one per SpacyToken, with different vectors (zu, dem)
        Type embeddingType = cas.getTypeSystem().getType(WORD_EMBEDDING);
        Feature reference = embeddingType.getFeatureByBaseName("reference");
        Feature embedding = embeddingType.getFeatureByBaseName("embedding");
        List<AnnotationFS> zum = new ArrayList<>();
        for (AnnotationFS e : select(cas, WORD_EMBEDDING)) {
            if (e.getCoveredText().equals("zum")) {
                zum.add(e);
            }
        }
        assertEquals(2, zum.size(), "a vector per word of zum");
        assertTrue(zum.get(0).getFeatureValue(reference) != zum.get(1).getFeatureValue(reference),
                "each references its own SpacyToken");
        float[] first = ((FloatArray) zum.get(0).getFeatureValue(embedding)).toArray();
        float[] second = ((FloatArray) zum.get(1).getFeatureValue(embedding)).toArray();
        assertFalse(Arrays.equals(first, second), "zu and dem have different vectors");

        // the given layers stay
        assertEquals(5, JCasUtil.select(cas, Token.class).stream()
                .filter(t -> t.getType().getName().equals(Token.class.getName())).count());
    }

    @Test
    public void writeTypesAsCommaSeparatedString() throws Exception {
        // only tokens (with their vectors): spaCy writes no dependencies, only Stanza's are in the CAS
        JCas cas = run(Map.of("write_types", Token.class.getName()));
        assertEquals(6, select(cas, SPACY_TOKEN).size());
        assertEquals(6, JCasUtil.select(cas, Dependency.class).size(), "only the Stanza dependencies");
        assertEquals(6, select(cas, WORD_EMBEDDING).size());
    }
}

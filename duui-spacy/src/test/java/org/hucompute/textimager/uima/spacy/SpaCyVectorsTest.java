package org.hucompute.textimager.uima.spacy;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.SpacyAnnotatorMetaData;
import org.texttechnologylab.uima.type.WordEmbedding;
import org.texttechnologylab.uima.type.spacy.SpacyToken;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Token vectors: {@link WordEmbedding} annotations (default), {@code SpacyToken.vector} ({@code vectors_in_spacytoken})
 * or none ({@code write_vectors=false}).
 *
 * <p>Needs a running service, URL from the system property {@code spacy.url} (default {@code http://localhost:9801}).
 */
public class SpaCyVectorsTest {

    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");

    // "Xqzvbrtl" has no vector in de_core_news_lg
    private static final String TEXT = "Die Katze schläft auf dem Sofa. Xqzvbrtl ist kein Wort.";

    private static JCas run(Map<String, String> parameters) throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(TEXT);
        cas.setDocumentLanguage("de");
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(SPACY_URL)
                .withParameter("model_variant", "large");
        parameters.forEach(component::withParameter);
        composer.add(component.build().withTimeout(1000L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
        return cas;
    }

    @Test
    public void wordEmbeddingsByDefault() throws Exception {
        JCas cas = run(Map.of());

        List<SpacyToken> tokens = new ArrayList<>(JCasUtil.select(cas, SpacyToken.class));
        List<WordEmbedding> embeddings = new ArrayList<>(JCasUtil.select(cas, WordEmbedding.class));
        assertFalse(embeddings.isEmpty());

        int withVector = 0;
        for (SpacyToken token : tokens) {
            assertNull(token.getVector(), "SpacyToken.vector of " + token.getCoveredText());
            if (token.getHasVector()) {
                withVector++;
            }
        }
        assertEquals(withVector, embeddings.size(), "one WordEmbedding per token with a vector");
        assertTrue(withVector < tokens.size(), "a token without vector");

        for (WordEmbedding embedding : embeddings) {
            SpacyToken token = (SpacyToken) embedding.getReference();
            assertNotNull(token);
            assertTrue(token.getHasVector());
            assertEquals(token.getBegin(), embedding.getBegin());
            assertEquals(token.getEnd(), embedding.getEnd());
            assertEquals(token.getIsOov(), embedding.getOov());
            assertEquals(300, embedding.getEmbedding().size());
            assertEquals("SPACYlg", embedding.getShortName());
            assertEquals("de_core_news_lg-3.8.0", embedding.getModelReference().getSource());
            assertEquals("de", embedding.getModelReference().getLang());
            assertSame(embeddings.get(0).getModelReference(), embedding.getModelReference(), "one MetaData per document");
        }
        long meta = JCasUtil.select(cas, SpacyAnnotatorMetaData.class).stream()
                .filter(m -> m.getReference() instanceof WordEmbedding).count();
        assertEquals(embeddings.size(), meta, "meta data of the WordEmbeddings");
    }

    @Test
    public void shortNameParameter() throws Exception {
        JCas cas = run(Map.of("vectors_short_name", "MYVECTORS"));
        assertFalse(JCasUtil.select(cas, WordEmbedding.class).isEmpty());
        for (WordEmbedding embedding : JCasUtil.select(cas, WordEmbedding.class)) {
            assertEquals("MYVECTORS", embedding.getShortName());
        }
    }

    @Test
    public void vectorsInSpacyToken() throws Exception {
        JCas cas = run(Map.of("vectors_in_spacytoken", "true"));
        assertTrue(JCasUtil.select(cas, WordEmbedding.class).isEmpty());
        for (SpacyToken token : JCasUtil.select(cas, SpacyToken.class)) {
            assertNotNull(token.getVector(), "SpacyToken.vector of " + token.getCoveredText());
            assertEquals(300, token.getVector().size());
        }

        // same vectors as the WordEmbeddings of the default
        JCas byDefault = run(Map.of());
        for (WordEmbedding embedding : JCasUtil.select(byDefault, WordEmbedding.class)) {
            SpacyToken token = JCasUtil.selectAt(cas, SpacyToken.class, embedding.getBegin(), embedding.getEnd()).get(0);
            assertArrayEquals(embedding.getEmbedding().toArray(), token.getVector().toArray(), embedding.getCoveredText());
        }
    }

    @Test
    public void noVectors() throws Exception {
        for (String inSpacyToken : List.of("false", "true")) {
            JCas cas = run(Map.of("write_vectors", "false", "vectors_in_spacytoken", inSpacyToken));
            assertTrue(JCasUtil.select(cas, WordEmbedding.class).isEmpty());
            List<SpacyToken> tokens = new ArrayList<>(JCasUtil.select(cas, SpacyToken.class));
            assertFalse(tokens.isEmpty());
            for (SpacyToken token : tokens) {
                assertNull(token.getVector());
            }
        }
    }
}

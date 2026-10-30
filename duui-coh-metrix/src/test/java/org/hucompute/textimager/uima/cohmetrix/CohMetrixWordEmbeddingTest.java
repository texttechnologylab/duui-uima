package org.hucompute.textimager.uima.cohmetrix;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FloatArray;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.WordEmbedding;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Token vectors from {@link WordEmbedding} annotations (README, "Vectors"): selected by their {@code shortName}, which
 * becomes the label group, and the same values as the old {@code SpacyToken.vector} input.
 *
 * <p>Needs running services: duui-spacy 0.8.0 (system property {@code spacy.url}, default
 * {@code http://localhost:9801}) and Coh-Metrix ({@code cohmetrix.url}, default {@code http://localhost:9802}).
 */
public class CohMetrixWordEmbeddingTest {

    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");
    private static final String COH_METRIX_URL = System.getProperty("cohmetrix.url", "http://localhost:9802");

    private static final String INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index";

    private static final List<String> VECTOR_INDICES = List.of(
            "LSASS1", "LSASS1d", "LSASSp", "LSASSpd", "LSAPP1", "LSAPP1d", "LSAGN", "LSAGNd", "SMCAUSlsa");

    private static final String TEXT = "Die Katze schläft auf dem Sofa. Der Hund liegt daneben. Beide Tiere sind müde.\n\n"
            + "Am Abend gehen die Tiere nach draußen. Die Katze jagt eine Maus. Der Hund bellt laut.";

    private static TypeSystemDescription typeSystem;

    @BeforeAll
    public static void typeSystem() throws Exception {
        // the CAS needs the type system of the components (Coh-Metrix index types)
        DUUIComposer composer = cohMetrix(Map.of());
        try {
            typeSystem = composer.instantiate_pipeline();
        } finally {
            composer.shutdown();
        }
    }

    @AfterAll
    public static void done() {
        typeSystem = null;
    }

    private static DUUIComposer composer(String url, Map<String, String> parameters) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(url);
        parameters.forEach(component::withParameter);
        composer.add(component.build().withTimeout(1000L));
        return composer;
    }

    private static DUUIComposer cohMetrix(Map<String, String> parameters) throws Exception {
        Map<String, String> withVectorIndices = new TreeMap<>(parameters);
        withVectorIndices.put("include_indices", String.join(",", VECTOR_INDICES));
        return composer(COH_METRIX_URL, withVectorIndices);
    }

    private static void run(DUUIComposer composer, JCas cas) throws Exception {
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    /** German text with two paragraphs, processed by spaCy with the given parameters. */
    private static JCas spacy(Map<String, String> parameters) throws Exception {
        JCas cas = JCasFactory.createJCas(typeSystem);
        cas.setDocumentText(TEXT);
        cas.setDocumentLanguage("de");
        int begin = 0;
        for (String paragraph : TEXT.split("\n\n", -1)) {
            new Paragraph(cas, begin, begin + paragraph.length()).addToIndexes();
            begin += paragraph.length() + 2;
        }
        Map<String, String> withModel = new TreeMap<>(parameters);
        withModel.put("model_variant", "large");
        run(composer(SPACY_URL, withModel), cas);
        return cas;
    }

    /** label_v3 -> {label_ttlab, value, error} of the Coh-Metrix indices. */
    private static Map<String, Object[]> indices(JCas cas) {
        CAS c = cas.getCas();
        Type type = c.getTypeSystem().getType(INDEX_TYPE);
        Map<String, Object[]> indices = new TreeMap<>();
        for (FeatureStructure fs : c.select(type).asList()) {
            indices.put(string(fs, "labelV3"), new Object[]{
                    string(fs, "labelTTLab"),
                    fs.getDoubleValue(fs.getType().getFeatureByBaseName("value")),
                    string(fs, "error")});
        }
        assertEquals(VECTOR_INDICES.size(), indices.size(), "indices: " + indices.keySet());
        return indices;
    }

    private static String string(FeatureStructure fs, String feature) {
        return fs.getStringValue(fs.getType().getFeatureByBaseName(feature));
    }

    private static Map<String, Object[]> computed(JCas cas, String group) {
        Map<String, Object[]> indices = indices(cas);
        for (Map.Entry<String, Object[]> index : indices.entrySet()) {
            assertEquals(index.getKey() + "_EMB_" + group, index.getValue()[0]);
            assertNull(index.getValue()[2], index.getKey() + " error");
            assertFalse(Double.isNaN((Double) index.getValue()[1]), index.getKey() + " value");
        }
        return indices;
    }

    private static void assertSameValues(Map<String, Object[]> expected, Map<String, Object[]> actual) {
        for (String label : VECTOR_INDICES) {
            assertEquals((Double) expected.get(label)[1], (Double) actual.get(label)[1], 0, label);
        }
    }

    private static void assertErrors(JCas cas, String errorPart) {
        for (Map.Entry<String, Object[]> index : indices(cas).entrySet()) {
            assertTrue(Double.isNaN((Double) index.getValue()[1]), index.getKey() + " value");
            assertNotNull(index.getValue()[2], index.getKey() + " error");
            assertTrue(((String) index.getValue()[2]).contains(errorPart), index.getValue()[2] + "");
        }
    }

    /** Copies every WordEmbedding with another shortName (vectors of a second vector space). */
    private static void copyEmbeddings(JCas cas, String shortName) {
        for (WordEmbedding embedding : new ArrayList<>(JCasUtil.select(cas, WordEmbedding.class))) {
            WordEmbedding copy = new WordEmbedding(cas, embedding.getBegin(), embedding.getEnd());
            FloatArray vector = new FloatArray(cas, embedding.getEmbedding().size());
            vector.copyFromArray(embedding.getEmbedding().toArray(), 0, 0, vector.size());
            copy.setEmbedding(vector);
            copy.setReference(embedding.getReference());
            copy.setShortName(shortName);
            copy.addToIndexes();
        }
    }

    @Test
    public void wordEmbeddingsEqualSpacyTokenVectors() throws Exception {
        JCas wordEmbeddings = spacy(Map.of());
        assertFalse(JCasUtil.select(wordEmbeddings, WordEmbedding.class).isEmpty());
        run(cohMetrix(Map.of()), wordEmbeddings);
        Map<String, Object[]> expected = computed(wordEmbeddings, "SPACYlg");

        // old input: vectors in SpacyToken, label from the spaCy model
        JCas spacyTokens = spacy(Map.of("vectors_in_spacytoken", "true"));
        assertTrue(JCasUtil.select(spacyTokens, WordEmbedding.class).isEmpty());
        run(cohMetrix(Map.of()), spacyTokens);
        assertSameValues(expected, computed(spacyTokens, "SPACYlg"));
    }

    @Test
    public void shortNameIsTheLabelGroup() throws Exception {
        JCas cas = spacy(Map.of("vectors_short_name", "MYVECTORS"));
        run(cohMetrix(Map.of()), cas);
        computed(cas, "MYVECTORS");
    }

    @Test
    public void parameterSelectsTheVectors() throws Exception {
        JCas reference = spacy(Map.of());
        run(cohMetrix(Map.of()), reference);
        Map<String, Object[]> expected = computed(reference, "SPACYlg");

        // two vector spaces: without the parameter an error naming both, with it the chosen one
        JCas ambiguous = spacy(Map.of());
        copyEmbeddings(ambiguous, "COPY");
        run(cohMetrix(Map.of()), ambiguous);
        assertErrors(ambiguous, "SPACYlg,COPY");

        JCas chosen = spacy(Map.of());
        copyEmbeddings(chosen, "COPY");
        run(cohMetrix(Map.of("vectors", "COPY")), chosen);
        assertSameValues(expected, computed(chosen, "COPY"));

        // a name not in the CAS
        JCas missing = spacy(Map.of());
        run(cohMetrix(Map.of("vectors", "OTHER")), missing);
        assertErrors(missing, "No token vectors (OTHER)");
    }

    @Test
    public void noVectors() throws Exception {
        // write_vectors=false: neither WordEmbedding nor SpacyToken.vector (SpacyToken.hasVector is still set)
        JCas cas = spacy(Map.of("write_vectors", "false"));
        run(cohMetrix(Map.of()), cas);
        assertErrors(cas, "No token vectors (SPACYlg)");
    }
}

package org.hucompute.textimager.uima.cohmetrix;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Best effort without UD (README, "Best effort without UD"): spaCy-only input (TIGER / ClearNLP labels) is computed, the
 * records that read the dependency relations name the mapping, the Text Easability components are None with an error.
 *
 * <p>Needs running services: duui-spacy (system property {@code spacy.url}, default {@code http://localhost:9801}) and
 * Coh-Metrix ({@code cohmetrix.url}, default {@code http://localhost:9802}).
 */
public class CohMetrixBestEffortTest {

    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");
    private static final String COH_METRIX_URL = System.getProperty("cohmetrix.url", "http://localhost:9802");
    private static final String INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index";

    // records that read the dependency relations, and some that do not
    private static final Set<String> MAPPED = Set.of("SYNNP", "SYNLE", "SYNSTRUTa", "DRPVAL", "SMTEMP");
    private static final Set<String> NOT_MAPPED = Set.of("DESWC", "DESSC", "WRDNOUN", "LDTTRc");
    private static final Set<String> PC = Set.of("PCNARz", "PCSYNz");

    private static TypeSystemDescription typeSystem;

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

    @BeforeAll
    public static void typeSystem() throws Exception {
        DUUIComposer composer = composer(COH_METRIX_URL, Map.of());
        try {
            typeSystem = composer.instantiate_pipeline();
        } finally {
            composer.shutdown();
        }
    }

    /** label_v3 -> its records {label_ttlab, value, error} after spaCy (own tokens, sentences, parse) and Coh-Metrix. */
    private static Map<String, List<Object[]>> run(String language, String text) throws Exception {
        JCas cas = JCasFactory.createJCas(typeSystem);
        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);
        new Paragraph(cas, 0, text.length()).addToIndexes();
        DUUIComposer composer = composer(SPACY_URL, Map.of("model_variant", "large"));
        composer.add(new DUUIRemoteDriver.Component(COH_METRIX_URL)
                .withParameter("include_indices", String.join(",", MAPPED) + "," + String.join(",", NOT_MAPPED) + ","
                        + String.join(",", PC))
                .build().withTimeout(1000L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
        CAS c = cas.getCas();
        Type type = c.getTypeSystem().getType(INDEX_TYPE);
        Map<String, List<Object[]>> indices = new TreeMap<>();
        for (FeatureStructure fs : c.select(type).asList()) {
            indices.computeIfAbsent(string(fs, "labelV3"), k -> new ArrayList<>()).add(new Object[]{string(fs, "labelTTLab"),
                    fs.getDoubleValue(fs.getType().getFeatureByBaseName("value")), string(fs, "error")});
        }
        return indices;
    }

    private static String string(FeatureStructure fs, String feature) {
        return fs.getStringValue(fs.getType().getFeatureByBaseName(feature));
    }

    private static void assertBestEffort(Map<String, List<Object[]>> indices, String group) {
        assertEquals(MAPPED.size() + NOT_MAPPED.size() + PC.size(), indices.size(), "indices: " + indices.keySet());
        for (String label : MAPPED) {
            for (Object[] record : indices.get(label)) {
                assertTrue(((String) record[0]).startsWith(label + "_" + group), (String) record[0]);
                assertFalse(Double.isNaN((Double) record[1]), record[0] + " is computed");
            }
        }
        for (String label : NOT_MAPPED) {
            for (Object[] record : indices.get(label)) {
                assertFalse(((String) record[0]).contains("map"), (String) record[0]);
                assertFalse(Double.isNaN((Double) record[1]), record[0] + " is computed");
            }
        }
        // the main model: None with the reason; the CLEAR model is None anyway (not refitted yet)
        for (String label : PC) {
            for (Object[] record : indices.get(label)) {
                assertTrue(Double.isNaN((Double) record[1]), (String) record[0]);
                assertNotNull(record[2], (String) record[0]);
            }
            assertTrue(indices.get(label).stream().anyMatch(r -> ((String) r[2]).contains("mapped")), label);
        }
    }

    @Test
    public void germanTiger() throws Exception {
        assertBestEffort(run("de", "Der Kuchen wurde am Abend gegessen. Die Großmutter hatte ihn gebacken, weil Anna kam. "
                + "Sie würde gern ins Kino gehen."), "TIGERmap");
    }

    @Test
    public void englishClearNlp() throws Exception {
        assertBestEffort(run("en", "The cake was eaten in the evening. Her grandmother had baked it because Anna came. "
                + "The house is too small for the whole family."), "CLEARNLPmap");
    }
}

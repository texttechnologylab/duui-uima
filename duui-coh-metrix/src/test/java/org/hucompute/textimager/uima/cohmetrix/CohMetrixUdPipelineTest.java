package org.hucompute.textimager.uima.cohmetrix;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * The default pipeline (README, "Prerequisites and preprocessing"), the DUUI form of the reference preprocessing of
 * Coh-Metrix 0.4.0-py (its defaults per language):
 * <ol>
 *     <li>paragraphs: duui-chonky {@code keep_line_breaks=filter} (de) or duui-wtpsplit {@code segment=paragraphs}
 *     (en);</li>
 *     <li>sentences: duui-wtpsplit within each paragraph, {@code split_on_input_newlines=true};</li>
 *     <li>tokens: duui-somajo within each sentence;</li>
 *     <li>duui-stanza: StanzaToken, StanzaSentence, POS, lemma, morphology, UD dependencies;</li>
 *     <li>duui-spacy {@code token_type=StanzaToken}, {@code write_types=Token}: only a SpacyToken and a WordEmbedding per
 *     Stanza word;</li>
 *     <li>duui-remove-annotations: the plain Sentence and Token (exact type);</li>
 *     <li>Coh-Metrix.</li>
 * </ol>
 *
 * <p>Needs the running services, URLs as system properties: {@code chonky.url}, {@code wtpsplit.url},
 * {@code somajo.url}, {@code stanza.url}, {@code spacy.url}, {@code remove.url}, {@code cohmetrix.url} (defaults
 * below).
 */
public class CohMetrixUdPipelineTest {

    private static final String CHONKY_URL = System.getProperty("chonky.url", "http://localhost:9805");
    private static final String WTPSPLIT_URL = System.getProperty("wtpsplit.url", "http://localhost:9804");
    private static final String SOMAJO_URL = System.getProperty("somajo.url", "http://localhost:9807");
    private static final String STANZA_URL = System.getProperty("stanza.url", "http://localhost:9806");
    private static final String SPACY_URL = System.getProperty("spacy.url", "http://localhost:9801");
    private static final String REMOVE_URL = System.getProperty("remove.url", "http://localhost:9808");
    private static final String COH_METRIX_URL = System.getProperty("cohmetrix.url", "http://localhost:9802");

    private static final String STANZA_TOKEN = "org.texttechnologylab.uima.type.stanza.StanzaToken";
    private static final String STANZA_SENTENCE = "org.texttechnologylab.uima.type.stanza.StanzaSentence";
    private static final String SPACY_TOKEN = "org.texttechnologylab.uima.type.spacy.SpacyToken";
    private static final String WORD_EMBEDDING = "org.texttechnologylab.uima.type.WordEmbedding";
    private static final String PARAGRAPH = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph";
    private static final String INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index";

    // indices that read the dependency relations (SYNNP, DRPVAL), the vectors (LSASS1) and neither (DESWC, DESSC)
    private static final Set<String> INDICES = Set.of("DESWC", "DESSC", "SYNNP", "DRPVAL", "LSASS1");

    private static final long TIMEOUT = 1000L;

    private static TypeSystemDescription typeSystem;

    private static DUUIRemoteDriver.Component remote(String url) throws Exception {
        return new DUUIRemoteDriver.Component(url);
    }

    private static DUUIComposer composer() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        return composer;
    }

    @BeforeAll
    public static void typeSystem() throws Exception {
        DUUIComposer composer = composer();
        composer.add(remote(STANZA_URL).build().withTimeout(TIMEOUT));
        composer.add(remote(COH_METRIX_URL).build().withTimeout(TIMEOUT));
        try {
            typeSystem = composer.instantiate_pipeline();
        } finally {
            composer.shutdown();
        }
    }

    /** The Coh-Metrix service of the default pipeline ({@code cohmetrix.url}). */
    public static DUUIRemoteDriver.Component cohMetrix() throws Exception {
        return remote(COH_METRIX_URL);
    }

    /**
     * The default pipeline after the paragraphs (steps 2 to 7 of the class comment, the services of the system
     * properties): sentences, tokens, Stanza, spaCy vectors, removal of the plain layers, then {@code cohMetrix}. Also
     * used for corpora with their own paragraphs (CLEAR, {@code ClearCorpusCohMetrixTest}).
     */
    public static void addAfterParagraphs(DUUIComposer composer, DUUIRemoteDriver.Component cohMetrix)
            throws Exception {
        composer.add(remote(WTPSPLIT_URL).withParameter("split_on_input_newlines", "true").build()
                .withTimeout(TIMEOUT));
        composer.add(remote(SOMAJO_URL).build().withTimeout(TIMEOUT));
        composer.add(remote(STANZA_URL).build().withTimeout(TIMEOUT));
        composer.add(remote(SPACY_URL)
                .withParameter("model_variant", "large")
                .withParameter("use_existing_tokens", "true")
                .withParameter("use_existing_sentences", "true")
                .withParameter("token_type", STANZA_TOKEN)
                // only the SpacyTokens (with their WordEmbeddings): the Stanza layer has POS, lemmas, morphology and
                // dependencies already, Coh-Metrix reads only the vectors from spaCy
                .withParameter("write_types", Token.class.getName())
                .build().withTimeout(TIMEOUT));
        composer.add(remote(REMOVE_URL)
                .withParameter("types", Sentence.class.getName() + "," + Token.class.getName())
                .withParameter("exact", "true")
                .build().withTimeout(TIMEOUT));
        composer.add(cohMetrix.build().withTimeout(TIMEOUT));
    }

    /** The CAS after the default pipeline for {@code language} (de or en). */
    static JCas run(String language, String text, String includeIndices) throws Exception {
        JCas cas = JCasFactory.createJCas(typeSystem);
        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);
        DocumentMetaData meta = DocumentMetaData.create(cas);
        meta.setDocumentId("pipeline-" + language);

        DUUIComposer composer = composer();
        if (language.equals("de")) {
            composer.add(remote(CHONKY_URL).withParameter("keep_line_breaks", "filter").build().withTimeout(TIMEOUT));
        } else {
            composer.add(remote(WTPSPLIT_URL).withParameter("segment", "paragraphs").build().withTimeout(TIMEOUT));
        }
        DUUIRemoteDriver.Component cohMetrix = remote(COH_METRIX_URL);
        if (includeIndices != null) {
            cohMetrix.withParameter("include_indices", includeIndices);
        }
        addAfterParagraphs(composer, cohMetrix);
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
        return cas;
    }

    private static List<Annotation> exactly(JCas cas, String typeName) {
        List<Annotation> out = new ArrayList<>();
        for (Annotation a : cas.getAnnotationIndex()) {
            if (a.getType().getName().equals(typeName)) {
                out.add(a);
            }
        }
        return out;
    }

    private static String string(FeatureStructure fs, String feature) {
        return fs.getStringValue(fs.getType().getFeatureByBaseName(feature));
    }

    private static void assertPipeline(JCas cas, int minParagraphs, Map<String, Integer> wordsOverToken) {
        String text = cas.getDocumentText();
        List<Annotation> paragraphs = exactly(cas, PARAGRAPH);
        assertTrue(paragraphs.size() >= minParagraphs, "paragraphs: " + paragraphs.size());
        for (Annotation p : paragraphs) {
            // every paragraph after the first starts a line (the line-break filter)
            assertTrue(p.getBegin() == 0 || text.substring(Math.max(0, p.getBegin() - 6), p.getBegin()).contains("\n"),
                    "paragraph at " + p.getBegin());
        }
        // the plain layers are removed, the Stanza and spaCy layers stay
        assertEquals(0, exactly(cas, Token.class.getName()).size(), "plain Tokens");
        assertEquals(0, exactly(cas, Sentence.class.getName()).size(), "plain Sentences");
        assertFalse(exactly(cas, STANZA_SENTENCE).isEmpty());
        List<Annotation> words = exactly(cas, STANZA_TOKEN);
        assertEquals(words.size(), exactly(cas, SPACY_TOKEN).size(), "a SpacyToken per Stanza word");
        assertEquals(words.size(), exactly(cas, WORD_EMBEDDING).size(), "a WordEmbedding per Stanza word");
        // spaCy writes no parse of its own (write_types): one dependency (incl. ROOT) per word, the Stanza ones
        assertEquals(words.size(), JCasUtil.select(cas, Dependency.class).size(), "dependencies");
        // split contractions: several words over the span of one token
        for (Map.Entry<String, Integer> e : wordsOverToken.entrySet()) {
            int begin = text.indexOf(e.getKey());
            long n = words.stream().filter(w -> w.getBegin() == begin && w.getEnd() == begin + e.getKey().length())
                    .count();
            assertEquals((long) e.getValue(), n, "words over \"" + e.getKey() + "\"");
        }

        // UD input: no mapping group; every selected index computed, the LSA ones with the spaCy lg vectors
        Map<String, List<String[]>> indices = new TreeMap<>();
        for (FeatureStructure fs : cas.getCas().select(cas.getTypeSystem().getType(INDEX_TYPE)).asList()) {
            double value = fs.getDoubleValue(fs.getType().getFeatureByBaseName("value"));
            indices.computeIfAbsent(string(fs, "labelV3"), k -> new ArrayList<>()).add(new String[]{
                    string(fs, "labelTTLab"), String.valueOf(value), string(fs, "error")});
        }
        assertEquals(INDICES, indices.keySet());
        for (List<String[]> records : indices.values()) {
            for (String[] record : records) {
                assertFalse(record[0].contains("map"), record[0]);
                assertFalse(Double.isNaN(Double.parseDouble(record[1])), record[0] + ": " + record[2]);
            }
        }
        assertTrue(indices.get("LSASS1").stream().allMatch(r -> r[0].contains("SPACYlg")), "LSASS1 vectors");
    }

    @Test
    public void german() throws Exception {
        String text = "Der Kuchen wurde am Abend im Garten gegessen. Die Großmutter hatte ihn am Morgen gebacken.\n"
                + "Zum Glück blieb noch ein Stück übrig.\n"
                + "Im nächsten Jahr will sie wieder backen, und zwar am liebsten im Sommer.";
        JCas cas = run("de", text, String.join(",", INDICES));
        // contractions split into two words, capitalised too
        assertPipeline(cas, 1, Map.of("Zum", 2, "Im", 2));
        // "am" before "Abend" is split, "am" before the superlative "liebsten" is not (STTS PTKA)
        String t = cas.getDocumentText();
        for (Map.Entry<String, Integer> e : Map.of("am Abend", 2, "am liebsten", 1, "im Garten", 2).entrySet()) {
            int begin = t.indexOf(e.getKey());
            assertEquals((long) e.getValue(), exactly(cas, STANZA_TOKEN).stream()
                    .filter(w -> w.getBegin() == begin && w.getEnd() == begin + 2).count(), e.getKey());
        }
    }

    @Test
    public void english() throws Exception {
        String text = "The cake was eaten in the evening. Her grandmother had baked it in the morning.\n"
                + "Luckily, there was still a piece left, and she didn't mind.\n"
                + "Next year she wants to bake again, preferably in the summer.";
        JCas cas = run("en", text, String.join(",", INDICES));
        // SoMaJo en_PTB splits "didn't" into the tokens "did" + "n't", so no word shares a span
        assertPipeline(cas, 1, Map.of("did", 1, "n't", 1));
    }
}

package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.spacy.SpacyToken;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Berkeley Neural Parser (benepar) in the spaCy component: parameter {@code use_benepar=true}, models benepar_en3 and
 * benepar_de2, written to {@link SpacyToken#getBeneparLabels()} (labels of the constituents that consist of exactly
 * this token; null if there is none).
 *
 * <p>Needs the spaCy service on {@code localhost:9714} (Docker image, or {@code uvicorn textimager_duui_spacy:app}),
 * as the other tests here. Uses {@code model_variant=large} (de_core_news_lg / en_core_web_lg). Memory: de_core_news_lg
 * with benepar_de2 needs up to ~6 GB while loading; with less RAM start the service with
 * {@code TEXTIMAGER_SPACY_MODEL_CACHE_SIZE=1}.
 */
public class SpaCyBeneparTest {
    static JCas cas;

    @BeforeAll
    static void beforeAll() throws Exception {
        cas = JCasFactory.createJCas();
    }

    @AfterEach
    public void afterEach() {
        cas.reset();
    }

    /**
     * One composer per call: after resetPipeline() a composer reused the component instantiated first (and its
     * parameters), so a run without benepar made the later runs of the same composer run without benepar as well.
     */
    private static void run(String language, String text, boolean useBenepar) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withParameter("use_benepar", String.valueOf(useBenepar))
                .withParameter("model_variant", "large")
                .build()
                // seconds; loading spaCy lg + benepar_de2 takes about a minute (DUUI's default is 60 s, after which it
                // sends the document again while the service is still loading)
                .withTimeout(600L));
        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    /** token text -> benepar labels (joined by "+"; null if the token has none), in text order */
    private static Map<String, String> labels(Collection<SpacyToken> tokens) {
        Map<String, String> labels = new LinkedHashMap<>();
        for (SpacyToken token : tokens) {
            String key = token.getCoveredText() + "@" + token.getBegin();
            labels.put(key, token.getBeneparLabels() == null ? null : String.join("+", token.getBeneparLabels().toArray()));
        }
        return labels;
    }

    @Test
    public void englishLabels() throws Exception {
        run("en", "This is an iPhone by Apple. And this is an iMac.", true);

        Map<String, String> labels = labels(JCasUtil.select(cas, SpacyToken.class));
        System.out.println(labels);
        assertEquals(2, JCasUtil.select(cas, Sentence.class).size());
        assertEquals(13, labels.size());
        // one-token constituents get a label, all other tokens none
        assertEquals("NP", labels.get("This@0"));
        assertEquals("NP", labels.get("Apple@21"));
        assertEquals("NP", labels.get("this@32"));
        assertEquals(3, labels.values().stream().filter(l -> l != null).count());
    }

    @Test
    public void germanRuns() throws Exception {
        String text = "Das ist ein iPhone von Apple. Und das ist ein iMac.";
        run("de", text, true);

        // benepar_de2 (SPMRL treebank) has no constituents of a single word, so the tokens get no labels; the test
        // checks that the request succeeds with benepar and returns all tokens and sentences
        Collection<SpacyToken> tokens = JCasUtil.select(cas, SpacyToken.class);
        assertEquals(13, tokens.size());
        assertEquals(2, JCasUtil.select(cas, Sentence.class).size());
        for (SpacyToken token : tokens) {
            assertNull(token.getBeneparLabels(), token.getCoveredText());
        }
    }

    /**
     * A sentence with more than 512 sub-word tokens cannot be parsed by benepar: it is skipped (its tokens get no
     * labels), the other sentences are parsed, and the response still has all tokens (before the fix the whole
     * response was empty).
     */
    @Test
    public void tooLongSentenceIsSkipped() throws Exception {
        for (String[] c : new String[][]{
                {"en", "This is an iPhone by Apple. I eat ", "apple", ". And this is an iMac."},
                {"de", "Das ist ein iPhone von Apple. Ich esse ", "Apfel", ". Und das ist ein iMac."}}) {
            List<String> items = new ArrayList<>();
            for (int i = 0; i < 300; i++) {
                items.add(c[2]);
            }
            String longSentence = c[1] + String.join(", ", items) + c[3];

            run(c[0], longSentence, false);
            int tokensWithoutBenepar = JCasUtil.select(cas, SpacyToken.class).size();
            int sentencesWithoutBenepar = JCasUtil.select(cas, Sentence.class).size();
            afterEach();

            run(c[0], longSentence, true);
            Collection<SpacyToken> tokens = JCasUtil.select(cas, SpacyToken.class);
            assertTrue(tokensWithoutBenepar > 600, c[0] + ": " + tokensWithoutBenepar);
            assertEquals(tokensWithoutBenepar, tokens.size(), c[0]);
            assertEquals(sentencesWithoutBenepar, JCasUtil.select(cas, Sentence.class).size(), c[0]);
            assertEquals(3, sentencesWithoutBenepar, c[0]);

            if (c[0].equals("en")) {
                Map<String, String> labels = labels(tokens);
                assertEquals("NP", labels.get("This@0"));
                assertEquals("NP", labels.get("Apple@21"));
                // the skipped sentence: no labels at all ("I" alone would be an NP)
                List<SpacyToken> skipped = JCasUtil.selectCovered(cas, SpacyToken.class,
                        JCasUtil.select(cas, Sentence.class).stream().skip(1).findFirst().orElseThrow());
                assertFalse(skipped.isEmpty());
                for (SpacyToken token : skipped) {
                    assertNull(token.getBeneparLabels(), token.getCoveredText() + "@" + token.getBegin());
                }
                // the sentence after it is parsed
                assertTrue(labels.entrySet().stream().anyMatch(e -> e.getKey().startsWith("this@") && "NP".equals(e.getValue())),
                        labels.toString());
            }
            afterEach();
        }
    }
}

package org.hucompute.textimager.uima.pos;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.Collection;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Integration tests for the Ancient Greek POS tagger DUUI component.
 */
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class AncientGreekPOSTest {

    private static DUUIComposer composer;
    private static final String ENDPOINT = "http://localhost:9714";

    @BeforeAll
    static void setUp() throws Exception {
        // Initialize DUUI composer with a remote driver
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        composer = new DUUIComposer()
                .withLuaContext(ctx)
                .withSkipVerification(true);

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        // Add the remote POS component
        composer.add(new DUUIRemoteDriver.Component(ENDPOINT));

        System.out.println("DUUI Composer initialized, endpoint: " + ENDPOINT);
    }

    @AfterAll
    static void tearDown() throws Exception {
        if (composer != null) {
            composer.shutdown();
        }
    }

    /**
     * Test 1: Simple single-line Ancient Greek sentence.
     * Verifies POS annotations are created and cover known words.
     */
    @Test
    @Order(1)
    @DisplayName("Test simple Iliad opening line")
    void testSimpleSentence() throws Exception {
        String text = "Μῆνιν ἄειδε θεὰ Πηληϊάδεω Ἀχιλῆος";

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(text);
        jCas.setDocumentLanguage("grc");

        // Run the pipeline
        composer.run(jCas);

        // Collect POS annotations
        Collection<POS> posAnnotations = JCasUtil.select(jCas, POS.class);

        System.out.println("\n--- Test 1: Simple Sentence ---");
        System.out.println("Text: " + text);
        System.out.println("POS annotations found: " + posAnnotations.size());

        for (POS pos : posAnnotations) {
            String word = pos.getCoveredText();
            System.out.printf("  [%d:%d] %-20s → %s%n",
                    pos.getBegin(), pos.getEnd(), word, pos.getPosValue());
        }

        // Assertions
        assertFalse(posAnnotations.isEmpty(),
                "Should have at least one POS annotation");
        assertTrue(posAnnotations.size() >= 5,
                "Expected at least 5 tokens, got " + posAnnotations.size());

        // Verify every annotation has a valid POS value
        for (POS pos : posAnnotations) {
            assertNotNull(pos.getPosValue(),
                    "POS value should not be null for: " + pos.getCoveredText());
            assertFalse(pos.getPosValue().isEmpty(),
                    "POS value should not be empty for: " + pos.getCoveredText());
            assertTrue(pos.getBegin() >= 0, "Begin offset should be >= 0");
            assertTrue(pos.getEnd() <= text.length(),
                    "End offset should be <= text length");
            assertTrue(pos.getBegin() < pos.getEnd(),
                    "Begin should be < End");
        }
    }

    /**
     * Test 2: Multi-line passage from the Iliad.
     * Verifies annotations cover the entire document.
     */
    @Test
    @Order(2)
    @DisplayName("Test multi-line Iliad passage")
    void testMultiLineSentence() throws Exception {
        String text =
                "οὐλομένην, ἣ μυρί' Ἀχαιοῖς ἄλγε' ἔθηκε\n" +
                        "πολλὰς δ' ἰφθίμους ψυχὰς Ἄϊδι προΐαψεν ,\n" +
                        "ἡρώων, αὐτοὺς δὲ ἑλώρια τεῦχε κύνεσσιν";

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(text);
        jCas.setDocumentLanguage("grc");

        composer.run(jCas);

        Collection<POS> posAnnotations = JCasUtil.select(jCas, POS.class);

        System.out.println("\n--- Test 2: Multi-line Passage ---");
        System.out.println("Text length: " + text.length());
        System.out.println("POS annotations found: " + posAnnotations.size());

        for (POS pos : posAnnotations) {
            String word = pos.getCoveredText();
            System.out.printf("  [%d:%d] %-20s → %s%n",
                    pos.getBegin(), pos.getEnd(), word, pos.getPosValue());
        }

        // Should have tokens from all three lines
        assertTrue(posAnnotations.size() >= 15,
                "Expected at least 15 tokens across 3 lines, got "
                        + posAnnotations.size());

        // Verify last annotation's end offset is within the text
        POS lastPos = null;
        for (POS pos : posAnnotations) {
            lastPos = pos;
        }
        assertNotNull(lastPos);
        assertTrue(lastPos.getEnd() <= text.length(),
                "Last token's end should be within text bounds");

        // Verify covered text matches the document
        for (POS pos : posAnnotations) {
            String covered = text.substring(pos.getBegin(), pos.getEnd());
            assertEquals(covered, pos.getCoveredText(),
                    "Covered text mismatch at offset " + pos.getBegin());
        }
    }

    /**
     * Test 3: Empty text input.
     * Verifies the component handles it gracefully without crashing.
     */
    @Test
    @Order(3)
    @DisplayName("Test empty text handling")
    void testEmptyText() throws Exception {
        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText("");
        jCas.setDocumentLanguage("grc");

        // Should not throw
        assertDoesNotThrow(() -> composer.run(jCas));

        Collection<POS> posAnnotations = JCasUtil.select(jCas, POS.class);

        System.out.println("\n--- Test 3: Empty Text ---");
        System.out.println("POS annotations found: " + posAnnotations.size());

        assertEquals(0, posAnnotations.size(),
                "Empty text should produce no POS annotations");
    }

    /**
     * Test 4: Verify specific POS tag for a known word.
     * "Μῆνιν" (wrath, accusative) should be tagged as NOUN.
     */
    @Test
    @Order(4)
    @DisplayName("Test known word POS prediction")
    void testKnownWordTag() throws Exception {
        String text = "Μῆνιν ἄειδε θεά";

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(text);
        jCas.setDocumentLanguage("grc");

        composer.run(jCas);

        Collection<POS> posAnnotations = JCasUtil.select(jCas, POS.class);

        System.out.println("\n--- Test 4: Known Word POS ---");
        for (POS pos : posAnnotations) {
            System.out.printf("  %-20s → %s%n",
                    pos.getCoveredText(), pos.getPosValue());
        }

        // Find the first token (should be Μῆνιν)
        POS firstToken = posAnnotations.iterator().next();
        assertEquals("Μῆνιν", firstToken.getCoveredText());
        assertEquals("NOUN", firstToken.getPosValue(),
                "Μῆνιν (wrath/acc) should be tagged as NOUN");

        // Verify ἄειδε is VERB
        boolean foundVerb = false;
        for (POS pos : posAnnotations) {
            if ("ἄειδε".equals(pos.getCoveredText())) {
                assertEquals("VERB", pos.getPosValue(),
                        "ἄειδε (sing!) should be tagged as VERB");
                foundVerb = true;
            }
        }
        assertTrue(foundVerb, "Should find ἄειδε in annotations");
    }
}
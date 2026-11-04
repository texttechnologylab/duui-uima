package org.texttechnologylab.duui.stanza;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.ROOT;
import org.apache.uima.cas.Type;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasUtil;
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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Needs a running service, URL from the system property {@code stanza.url} (default {@code http://localhost:9806}).
 * The Stanza types are read through the CAS API, so the test does not need their JCas classes.
 */
public class StanzaTest {

    private static final String STANZA_URL = System.getProperty("stanza.url", "http://localhost:9806");
    private static final String STANZA_TOKEN = "org.texttechnologylab.uima.type.stanza.StanzaToken";
    private static final String STANZA_SENTENCE = "org.texttechnologylab.uima.type.stanza.StanzaSentence";

    private static TypeSystemDescription typeSystem;

    @BeforeAll
    public static void typeSystem() throws Exception {
        // the CAS needs the type system of the component (StanzaToken, StanzaSentence)
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component(STANZA_URL).build());
        try {
            typeSystem = composer.instantiate_pipeline();
        } finally {
            composer.shutdown();
        }
    }

    private static void run(JCas cas) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component(STANZA_URL).build().withTimeout(600L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    /** One plain Sentence over the whole text, and a plain Token per whitespace-separated piece, with the sentence-final
     * punctuation as a token of its own (as an upstream tokenizer would give them). */
    private static JCas cas(String text, String language) throws Exception {
        JCas cas = JCasFactory.createJCas(typeSystem);
        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);
        new Sentence(cas, 0, text.length()).addToIndexes();
        int i = 0;
        while (i < text.length()) {
            while (i < text.length() && Character.isWhitespace(text.charAt(i))) i++;
            int begin = i;
            while (i < text.length() && !Character.isWhitespace(text.charAt(i))) i++;
            if (begin == i) break;
            int end = i;
            if (end - begin > 1 && ".,!?".indexOf(text.charAt(end - 1)) >= 0) {
                new Token(cas, begin, end - 1).addToIndexes();
                new Token(cas, end - 1, end).addToIndexes();
            } else {
                new Token(cas, begin, end).addToIndexes();
            }
        }
        return cas;
    }

    private static List<Token> stanzaTokens(JCas cas) {
        Type type = cas.getTypeSystem().getType(STANZA_TOKEN);
        assertNotNull(type, "StanzaToken type");
        List<Token> tokens = new ArrayList<>();
        for (Annotation a : CasUtil.select(cas.getCas(), type).stream().map(a -> (Annotation) a).toList()) {
            tokens.add((Token) a);
        }
        return tokens;
    }

    private static List<Sentence> stanzaSentences(JCas cas) {
        Type type = cas.getTypeSystem().getType(STANZA_SENTENCE);
        assertNotNull(type, "StanzaSentence type");
        List<Sentence> sentences = new ArrayList<>();
        for (Object a : CasUtil.select(cas.getCas(), type)) {
            sentences.add((Sentence) a);
        }
        return sentences;
    }

    private static long plainCount(JCas cas, Class<? extends Annotation> clazz) {
        return JCasUtil.select(cas, clazz).stream().filter(a -> a.getType().getName().equals(clazz.getName())).count();
    }

    private static String word(Token token) {
        return token.getForm() != null ? token.getForm().getValue() : token.getCoveredText();
    }

    @Test
    public void germanContractionsAndAnnotations() throws Exception {
        JCas cas = cas("Er geht zum Haus und sitzt im Garten am besten .", "de");
        run(cas);

        List<String> words = new ArrayList<>();
        for (Token token : stanzaTokens(cas)) {
            words.add(word(token));
        }
        // zum and im split into two words over the span of the token, "am besten" stays one word (STTS PTKA)
        assertEquals(List.of("Er", "geht", "zu", "dem", "Haus", "und", "sitzt", "in", "dem", "Garten", "am", "besten", "."),
                words);
        for (Token token : stanzaTokens(cas)) {
            if (word(token).equals("zu") || word(token).equals("dem") && token.getCoveredText().equals("zum")) {
                assertEquals("zum", token.getCoveredText());
                assertNotNull(token.getForm());
            }
            if (token.getCoveredText().equals("am")) {
                assertNull(token.getForm(), "am before a superlative is not split");
            }
            assertNotNull(token.getPos(), "POS");
            assertNotNull(token.getPos().getCoarseValue(), "UPOS");
            assertNotNull(token.getPos().getPosValue(), "STTS");
            assertNotNull(token.getLemma(), "lemma");
        }

        // one dependency per word, one root per sentence, governors inside the sentence
        assertEquals(stanzaTokens(cas).size(), JCasUtil.select(cas, Dependency.class).size());
        assertEquals(1, JCasUtil.select(cas, ROOT.class).size());
        ROOT root = JCasUtil.select(cas, ROOT.class).iterator().next();
        assertEquals("--", root.getDependencyType());
        assertEquals(root.getGovernor(), root.getDependent());

        // the upstream layers stay, the Stanza sentence layer is complete
        assertEquals(1, plainCount(cas, Sentence.class));
        assertEquals(11, plainCount(cas, Token.class));
        assertEquals(1, stanzaSentences(cas).size());
    }

    @Test
    public void germanMorphologyKeepsPossessorNumberAndMood() throws Exception {
        JCas cas = cas("Unser Haus wäre schön .", "de");
        run(cas);
        String unser = null, waere = null;
        for (Token token : stanzaTokens(cas)) {
            if (token.getCoveredText().equals("Unser")) unser = token.getMorph().getValue();
            if (token.getCoveredText().equals("wäre")) waere = token.getMorph().getValue();
        }
        assertTrue(unser != null && unser.contains("Number[psor]="), "Number[psor] in " + unser);
        assertTrue(waere != null && waere.contains("Mood=Sub"), "Mood=Sub in " + waere);
    }

    @Test
    public void englishUniversalDependencies() throws Exception {
        JCas cas = cas("He has read the book .", "en");
        run(cas);
        for (Dependency dependency : JCasUtil.select(cas, Dependency.class)) {
            String word = dependency.getDependent().getCoveredText();
            if (word.equals("has")) {
                assertEquals("aux", dependency.getDependencyType());
                assertEquals("read", dependency.getGovernor().getCoveredText());
                assertEquals("VBZ", dependency.getDependent().getPos().getPosValue());
            }
            if (word.equals("read")) {
                assertTrue(dependency instanceof ROOT, "the lexical verb is the UD root");
            }
        }
    }

    @Test
    public void offsetsAfterSurrogatePair() throws Exception {
        // the 𐍁 is a surrogate pair in Java: offsets after it must still be right
        JCas cas = cas("Das 𐍁 Zeichen ist im Text .", "de");
        run(cas);
        List<String> covered = new ArrayList<>();
        for (Token token : stanzaTokens(cas)) {
            covered.add(token.getCoveredText());
        }
        assertEquals(List.of("Das", "𐍁", "Zeichen", "ist", "im", "im", "Text", "."), covered);
    }

    @Test
    public void longSentenceIsParsedInPieces() throws Exception {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < 1200; i++) {
            text.append(i % 10 == 9 ? "und " : "Haus ");
        }
        text.append(".");
        JCas cas = cas(text.toString(), "de");
        run(cas);
        // 1201 tokens in pieces of at most 500: three Stanza sentences, the given sentence stays
        List<Sentence> sentences = stanzaSentences(cas);
        assertEquals(3, sentences.size());
        assertEquals(0, sentences.get(0).getBegin());
        assertEquals(cas.getDocumentText().length(), sentences.get(2).getEnd());
        assertEquals(1, plainCount(cas, Sentence.class));
        assertEquals(3, JCasUtil.select(cas, ROOT.class).size());
    }

    @Test
    public void unsupportedLanguageFails() throws Exception {
        JCas cas = cas("C'est un test .", "fr");
        assertThrows(Exception.class, () -> run(cas));
    }
}

package org.texttechnologylab.duui.somajo;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Needs a running service, URL from the system property {@code somajo.url} (default {@code http://localhost:9803}).
 */
public class SoMaJoTest {

    private static final String SOMAJO_URL = System.getProperty("somajo.url", "http://localhost:9803");

    private static void run(JCas cas) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        composer.add(new DUUIRemoteDriver.Component(SOMAJO_URL).build().withTimeout(60L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    private static List<String> tokens(JCas cas) {
        List<String> tokens = new ArrayList<>();
        for (Token token : JCasUtil.select(cas, Token.class)) {
            tokens.add(token.getCoveredText());
        }
        return tokens;
    }

    /** Sentence annotations for the given sentence texts, found in order in the document text. */
    private static void addSentences(JCas cas, String... sentences) {
        int from = 0;
        for (String sentence : sentences) {
            int begin = cas.getDocumentText().indexOf(sentence, from);
            new Sentence(cas, begin, begin + sentence.length()).addToIndexes();
            from = begin + sentence.length();
        }
    }

    @Test
    public void germanWithSentences() throws Exception {
        JCas cas = JCasFactory.createJCas();
        // the 𐍁 is a surrogate pair in Java: offsets after it must still be right
        cas.setDocumentText("Das ist z. B. ein Test   mit\nUmbruch und 𐍁 Zeichen. Er sagt: „Geh!“ - > Ende");
        cas.setDocumentLanguage("de");
        // the second "sentence" starts inside "Zeichen." on purpose: no token may cross it
        addSentences(cas, "Das ist z. B. ein Test   mit\nUmbruch und 𐍁 Zei", "chen. Er sagt: „Geh!“ - > Ende");
        run(cas);

        assertEquals(List.of("Das", "ist", "z.", "B.", "ein", "Test", "mit", "Umbruch", "und", "𐍁", "Zei",
                "chen", ".", "Er", "sagt", ":", "„", "Geh", "!", "“", "- >", "Ende"), tokens(cas));
        for (Token token : JCasUtil.select(cas, Token.class)) {
            assertEquals(1, JCasUtil.selectCovering(Sentence.class, token).size(), "token inside one sentence");
        }
    }

    @Test
    public void englishWholeTextWithoutSentences() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("Mr. Smith can't go to the U.S. office.\nIt's closed.");
        cas.setDocumentLanguage("en");
        run(cas);
        assertEquals(List.of("Mr.", "Smith", "ca", "n't", "go", "to", "the", "U.S.", "office", ".", "It", "'s",
                "closed", "."), tokens(cas));
        assertEquals(0, JCasUtil.select(cas, Sentence.class).size(), "no sentences written");
    }

    @Test
    public void noCamelCaseSplit() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("Das neue iPhone und die BahnCard.");
        cas.setDocumentLanguage("de");
        run(cas);
        assertTrue(tokens(cas).contains("BahnCard"), tokens(cas).toString());
        assertTrue(tokens(cas).contains("iPhone"), tokens(cas).toString());
    }

    @Test
    public void unsupportedLanguageFails() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("C'est un test.");
        cas.setDocumentLanguage("fr");
        assertThrows(Exception.class, () -> run(cas));
    }
}

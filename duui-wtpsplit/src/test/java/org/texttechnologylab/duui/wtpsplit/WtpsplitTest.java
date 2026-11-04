package org.texttechnologylab.duui.wtpsplit;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Needs a running service, URL from the system property {@code wtpsplit.url} (default {@code http://localhost:9804}).
 */
public class WtpsplitTest {

    private static final String WTPSPLIT_URL = System.getProperty("wtpsplit.url", "http://localhost:9804");

    // the 𐍁 is a surrogate pair in Java: offsets after it must still be right; the first paragraph has a sentence
    // that continues over a line break
    private static final String TEXT = "Das ist der erste Satz mit 𐍁 darin. Dieser Satz geht\nüber zwei Zeilen.\n\n"
            + "Der zweite Absatz hat nur einen Satz.";

    private static void run(JCas cas, String splitOnInputNewlines) throws Exception {
        run(cas, splitOnInputNewlines, null);
    }

    private static void run(JCas cas, String splitOnInputNewlines, String segment) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(WTPSPLIT_URL);
        if (splitOnInputNewlines != null) {
            component.withParameter("split_on_input_newlines", splitOnInputNewlines);
        }
        if (segment != null) {
            component.withParameter("segment", segment);
        }
        composer.add(component.build().withTimeout(600L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    private static JCas casWithParagraphs() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(TEXT);
        cas.setDocumentLanguage("de");
        int begin = 0;
        for (String paragraph : TEXT.split("\n\n")) {
            begin = TEXT.indexOf(paragraph, begin);
            new Paragraph(cas, begin, begin + paragraph.length()).addToIndexes();
            begin += paragraph.length();
        }
        return cas;
    }

    private static List<String> sentences(JCas cas) {
        List<String> sentences = new ArrayList<>();
        for (Sentence sentence : JCasUtil.select(cas, Sentence.class)) {
            sentences.add(sentence.getCoveredText());
        }
        return sentences;
    }

    /** Sentences are trimmed, inside one paragraph, do not overlap, and cover all non-whitespace text. */
    private static void checkStructure(JCas cas) {
        String text = cas.getDocumentText();
        boolean[] covered = new boolean[text.length()];
        int previousEnd = 0;
        for (Sentence sentence : JCasUtil.select(cas, Sentence.class)) {
            String s = sentence.getCoveredText();
            assertFalse(s.isEmpty());
            assertEquals(s.strip(), s, "sentence not trimmed");
            assertTrue(sentence.getBegin() >= previousEnd, "sentences overlap");
            previousEnd = sentence.getEnd();
            if (JCasUtil.select(cas, Paragraph.class).size() > 0) {
                assertEquals(1, JCasUtil.selectCovering(Paragraph.class, sentence).size(), "inside one paragraph: " + s);
            }
            for (int i = sentence.getBegin(); i < sentence.getEnd(); i++) {
                covered[i] = true;
            }
        }
        for (int i = 0; i < text.length(); i++) {
            if (!Character.isWhitespace(text.charAt(i))) {
                assertTrue(covered[i], "not in a sentence: offset " + i);
            }
        }
    }

    @Test
    public void splitOnInputNewlinesByDefault() throws Exception {
        JCas cas = casWithParagraphs();
        run(cas, null);
        checkStructure(cas);
        List<String> sentences = sentences(cas);
        System.out.println(sentences);
        assertEquals(List.of("Das ist der erste Satz mit 𐍁 darin.", "Dieser Satz geht", "über zwei Zeilen.",
                "Der zweite Absatz hat nur einen Satz."), sentences);
    }

    @Test
    public void lineBreaksAsSpaces() throws Exception {
        JCas cas = casWithParagraphs();
        run(cas, "false");
        checkStructure(cas);
        List<String> sentences = sentences(cas);
        System.out.println(sentences);
        assertEquals(List.of("Das ist der erste Satz mit 𐍁 darin.", "Dieser Satz geht\nüber zwei Zeilen.",
                "Der zweite Absatz hat nur einen Satz."), sentences);
    }

    @Test
    public void wholeTextWithoutParagraphs() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("Ein Satz. Noch ein Satz.");
        cas.setDocumentLanguage("de");
        run(cas, null);
        checkStructure(cas);
        assertEquals(List.of("Ein Satz.", "Noch ein Satz."), sentences(cas));
    }

    @Test
    public void invalidParameterFails() throws Exception {
        assertThrows(Exception.class, () -> run(casWithParagraphs(), "yes"));
        assertThrows(Exception.class, () -> run(casWithParagraphs(), null, "words"));
    }

    private static List<Paragraph> paragraphs(String text) throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(text);
        cas.setDocumentLanguage("de");
        run(cas, null, "paragraphs");
        assertEquals(0, JCasUtil.select(cas, Sentence.class).size(), "no sentences in paragraph mode");
        List<Paragraph> paragraphs = new ArrayList<>(JCasUtil.select(cas, Paragraph.class));
        System.out.println(paragraphs.stream().map(Paragraph::getCoveredText).toList());
        // trimmed, in order, covering all non-whitespace text
        boolean[] covered = new boolean[text.length()];
        int previousEnd = 0;
        for (Paragraph paragraph : paragraphs) {
            assertEquals(paragraph.getCoveredText().strip(), paragraph.getCoveredText());
            assertTrue(paragraph.getBegin() >= previousEnd, "paragraphs overlap");
            previousEnd = paragraph.getEnd();
            for (int i = paragraph.getBegin(); i < paragraph.getEnd(); i++) {
                covered[i] = true;
            }
        }
        for (int i = 0; i < text.length(); i++) {
            assertTrue(Character.isWhitespace(text.charAt(i)) || covered[i], "not in a paragraph: offset " + i);
        }
        // a line break between two paragraphs, or among the 5 characters before the second (Coh-Metrix reference preprocessing)
        for (int i = 1; i < paragraphs.size(); i++) {
            int begin = paragraphs.get(i).getBegin();
            String between = text.substring(paragraphs.get(i - 1).getEnd(), begin);
            String before = text.substring(Math.max(paragraphs.get(i - 1).getEnd(), begin - 5), begin);
            assertTrue(between.contains("\n") || before.contains("\n"), paragraphs.get(i).getCoveredText());
        }
        return paragraphs;
    }

    @Test
    public void paragraphsAtLineBreaks() throws Exception {
        List<Paragraph> paragraphs = paragraphs(TEXT);
        assertEquals("Der zweite Absatz hat nur einen Satz.", paragraphs.get(paragraphs.size() - 1).getCoveredText());
        // without a line break there is no boundary to keep
        assertEquals(1, paragraphs(TEXT.replace("\n", " ")).size());
    }
}

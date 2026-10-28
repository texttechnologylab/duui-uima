package org.texttechnologylab.duui.chonky;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Needs a running service, URL from the system property {@code chonky.url} (default {@code http://localhost:9805}).
 */
public class ChonkyTest {

    private static final String CHONKY_URL = System.getProperty("chonky.url", "http://localhost:9805");

    // three topics; line breaks and a blank line inside the second topic; the 𐍁 is a surrogate pair in Java
    private static final String TEXT = "Die Bundesregierung hat heute ein neues Klimaschutzgesetz beschlossen. Es sieht "
            + "vor, dass die Emissionen bis 2030 um 65 Prozent sinken. Umweltverbände kritisierten das Gesetz.\n"
            + "Im Fußball hat Bayern München gestern gegen Dortmund gewonnen. Das Spiel\nendete 3 zu 1 𐍁.\n  \n"
            + "Der Trainer zeigte sich zufrieden.\n"
            + "Das Wetter: Morgen wird es sonnig bei bis zu 25 Grad. Am Wochenende ziehen Gewitter auf.";

    private static List<Paragraph> run(String text, String keepLineBreaks) throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText(text);
        cas.setDocumentLanguage("de");
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(CHONKY_URL);
        if (keepLineBreaks != null) {
            component.withParameter("keep_line_breaks", keepLineBreaks);
        }
        composer.add(component.build().withTimeout(600L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
        List<Paragraph> paragraphs = new ArrayList<>(JCasUtil.select(cas, Paragraph.class));
        checkStructure(text, paragraphs);
        System.out.println(keepLineBreaks + ": " + paragraphs.stream().map(Paragraph::getCoveredText).toList());
        return paragraphs;
    }

    /** Paragraphs are trimmed, do not overlap, and cover all non-whitespace text. */
    private static void checkStructure(String text, List<Paragraph> paragraphs) {
        boolean[] covered = new boolean[text.length()];
        int previousEnd = 0;
        for (Paragraph paragraph : paragraphs) {
            String p = paragraph.getCoveredText();
            assertFalse(p.isEmpty());
            assertEquals(p.strip(), p, "paragraph not trimmed");
            assertTrue(paragraph.getBegin() >= previousEnd, "paragraphs overlap");
            previousEnd = paragraph.getEnd();
            for (int i = paragraph.getBegin(); i < paragraph.getEnd(); i++) {
                covered[i] = true;
            }
        }
        for (int i = 0; i < text.length(); i++) {
            if (!Character.isWhitespace(text.charAt(i))) {
                assertTrue(covered[i], "not in a paragraph: offset " + i);
            }
        }
    }

    @Test
    public void topics() throws Exception {
        List<Paragraph> paragraphs = run(TEXT, null);
        assertTrue(paragraphs.size() >= 3, "at least the three topics");
        assertTrue(paragraphs.get(0).getCoveredText().startsWith("Die Bundesregierung"));
        assertTrue(paragraphs.get(paragraphs.size() - 1).getCoveredText().endsWith("Gewitter auf."));
    }

    @Test
    public void keepLines() throws Exception {
        for (Paragraph paragraph : run(TEXT, "line")) {
            assertFalse(paragraph.getCoveredText().contains("\n"), paragraph.getCoveredText());
        }
    }

    @Test
    public void keepBlankLines() throws Exception {
        Pattern blankLine = Pattern.compile("\n[^\\S\n]*\n");
        for (Paragraph paragraph : run(TEXT, "blank_line")) {
            assertFalse(blankLine.matcher(paragraph.getCoveredText()).find(), paragraph.getCoveredText());
        }
    }

    @Test
    public void longTextOverTheModelWindow() throws Exception {
        // more than the 1024 tokens of the model window (chonky's pipeline uses a stride)
        String text = (TEXT + " ").repeat(15).strip();
        List<Paragraph> paragraphs = run(text, null);
        assertTrue(paragraphs.size() >= 15, "paragraphs: " + paragraphs.size());
    }

    @Test
    public void emptyTextAndInvalidParameter() throws Exception {
        assertEquals(0, run("  \n ", null).size());
        assertThrows(Exception.class, () -> run(TEXT, "paragraph"));
    }
}

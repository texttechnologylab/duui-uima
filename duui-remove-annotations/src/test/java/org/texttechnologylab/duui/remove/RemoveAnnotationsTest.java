package org.texttechnologylab.duui.remove;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.DocumentModification;
import org.texttechnologylab.uima.type.spacy.SpacyToken;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Needs a running service, URL from the system property {@code remove.url} (default {@code http://localhost:9802}).
 */
public class RemoveAnnotationsTest {

    private static final String REMOVE_URL = System.getProperty("remove.url", "http://localhost:9802");

    /** "Ein Test." with 3 plain tokens, 3 SpacyTokens (subtype of Token) and 1 sentence. */
    private static JCas cas() throws Exception {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("Ein Test.");
        cas.setDocumentLanguage("de");
        int[][] tokens = {{0, 3}, {4, 8}, {8, 9}};
        for (int[] t : tokens) {
            new Token(cas, t[0], t[1]).addToIndexes();
            new SpacyToken(cas, t[0], t[1]).addToIndexes();
        }
        new Sentence(cas, 0, 9).addToIndexes();
        return cas;
    }

    private static void run(JCas cas, String types, String exact) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(REMOVE_URL);
        if (types != null) {
            component.withParameter("types", types);
        }
        if (exact != null) {
            component.withParameter("exact", exact);
        }
        composer.add(component.build().withTimeout(60L));
        try {
            composer.run(cas);
        } finally {
            composer.shutdown();
        }
    }

    private static long plainTokens(JCas cas) {
        return JCasUtil.select(cas, Token.class).stream().filter(t -> !(t instanceof SpacyToken)).count();
    }

    @Test
    public void exactTypeByDefault() throws Exception {
        JCas cas = cas();
        run(cas, Token.class.getName(), null);
        assertEquals(0, plainTokens(cas));
        assertEquals(3, JCasUtil.select(cas, SpacyToken.class).size());
        assertEquals(1, JCasUtil.select(cas, Sentence.class).size());
        String comment = JCasUtil.selectSingle(cas, DocumentModification.class).getComment();
        System.out.println(comment);
        assertTrue(comment.contains(Token.class.getName() + ": 3"), comment);
    }

    @Test
    public void withSubtypes() throws Exception {
        JCas cas = cas();
        run(cas, Token.class.getName(), "false");
        assertEquals(0, JCasUtil.select(cas, Token.class).size());
        assertEquals(1, JCasUtil.select(cas, Sentence.class).size());
    }

    @Test
    public void severalTypes() throws Exception {
        JCas cas = cas();
        run(cas, Token.class.getName() + ", " + Sentence.class.getName(), "true");
        assertEquals(0, plainTokens(cas));
        assertEquals(3, JCasUtil.select(cas, SpacyToken.class).size());
        assertEquals(0, JCasUtil.select(cas, Sentence.class).size());
    }

    @Test
    public void failsWithoutTypesOrUnknownType() {
        assertThrows(Exception.class, () -> run(cas(), null, null));
        assertThrows(Exception.class, () -> run(cas(), "org.example.NoSuchType", null));
        assertThrows(Exception.class, () -> run(cas(), Token.class.getName(), "yes"));
    }
}

package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class SpaCyMultiTest {
    @Test
    public void multiTestEn() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714")//,
                // DUUIRemoteDriver.class
        );

        String language = "en";

        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("This is an IPhone by Apple. And this is an iMac.");
        cas.setDocumentLanguage(language);

        composer.run(cas);

        composer.shutdown();

        int[][] tokens = new int[][]{
                new int[]{0, 4}, //This
                new int[]{5, 7}, //is
                new int[]{8, 10}, //an
                new int[]{11, 17}, //IPhone
                new int[]{18, 20}, //by
                new int[]{21, 26}, //Apple
                new int[]{26, 27}, //.
                new int[]{28, 31}, //And
                new int[]{32, 36}, //this
                new int[]{37, 39}, //is
                new int[]{40, 42}, //an
                new int[]{43, 47}, //iMac
                new int[]{47, 48} //.
        };

        int[][] casTokens = JCasUtil.select(cas, Token.class)
                .stream()
                .map(s -> new int[] { s.getBegin(), s.getEnd() })
                .toArray(int[][]::new);

        assertArrayEquals(tokens, casTokens);

        int[][] sents = new int[][]{new int[]{0, 27}, new int[]{28, 48}};

        int[][] casSents = JCasUtil.select(cas, Sentence.class)
                .stream()
                .map(s -> new int[]{s.getBegin(), s.getEnd()})
                .toArray(int[][]::new);

        assertArrayEquals(sents, casSents);

        String[] pos = new String[]{
                "DT", "VBZ", "DT", /*"NN"*/ "NNP", "IN", "NNP", ".",
                "CC", "DT", "VBZ", "DT", "NNP", "."
        };

        String[] casPos = JCasUtil.select(cas, POS.class)
                .stream()
                .map(POS::getPosValue)
                .toArray(String[]::new);

        assertArrayEquals(pos, casPos);

        String[] posCoarse = new String[]{
                "PRON", "AUX", "DET", "PROPN", "ADP", "PROPN", "PUNCT",
                "CCONJ", "PRON", "AUX", "DET", "PROPN", "PUNCT"
        };

        String[] casPosCoarse = JCasUtil.select(cas, POS.class)
                .stream()
                .map(POS::getCoarseValue)
                .toArray(String[]::new);

        assertArrayEquals(posCoarse, casPosCoarse);

        String[] deps = new String[]{
                "NSUBJ", "--", "DET", "ATTR", "PREP", "POBJ", "PUNCT", "CC", "NSUBJ", "--", "DET", "ATTR", "PUNCT"
        };
        String[] casDeps = JCasUtil.select(cas, Dependency.class)
                .stream()
                .map(Dependency::getDependencyType)
                .toArray(String[]::new);

        assertArrayEquals(deps, casDeps);

        String[] ents = new String[]{
                // version 3.1.1
                //"ORG", "ORG"
                // version 3.2.2
                "ORG"
        };

        String[] casEnts = JCasUtil.select(cas, NamedEntity.class)
                .stream()
                .map(NamedEntity::getValue)
                .toArray(String[]::new);

        assertArrayEquals(ents, casEnts);
    }

    @Test
    public void multiTestDe() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714"),
                DUUIRemoteDriver.class
        );

        String language = "de";

        JCas cas = JCasFactory.createJCas();
        cas.setDocumentText("Das ist ein IPhone von Apple. Und das ist ein iMac.");
        cas.setDocumentLanguage(language);

        composer.run(cas);

        composer.shutdown();

        int[][] tokens = new int[][]{
                new int[]{0, 3}, //Das
                new int[]{4, 7}, //ist
                new int[]{8, 11}, //ein
                new int[]{12, 18}, //IPhone
                new int[]{19, 22}, //von
                new int[]{23, 28}, //Apple
                new int[]{28, 29}, //.
                new int[]{30, 33}, //Und
                new int[]{34, 37}, //das
                new int[]{38, 41}, //ist
                new int[]{42, 45}, //ein
                new int[]{46, 50}, //iMac
                new int[]{50, 51} //.
        };

        int[][] casTokens = JCasUtil.select(cas, Token.class)
                .stream()
                .map(s -> new int[]{s.getBegin(), s.getEnd()})
                .toArray(int[][]::new);

        assertArrayEquals(tokens, casTokens);

        int[][] sents = new int[][]{
                new int[]{0, 29},
                new int[]{30, 51}
        };

        int[][] casSents = JCasUtil.select(cas, Sentence.class)
                .stream()
                .map(s -> new int[]{s.getBegin(), s.getEnd()})
                .toArray(int[][]::new);

        assertArrayEquals(sents, casSents);

        String[] pos = new String[]{
                "PDS", "VAFIN", "ART", "NN", "APPR", "NE", "$.",
                "KON", "PDS", "VAFIN", "ART", "NN", "$."
        };

        String[] casPos = JCasUtil.select(cas, POS.class)
                .stream()
                .map(POS::getPosValue)
                .toArray(String[]::new);

        assertArrayEquals(pos, casPos);

        String[] posCoarse = new String[]{
                "PRON", "AUX", "DET", "NOUN", "ADP", "PROPN", "PUNCT",
                "CCONJ", "PRON", "AUX", "DET", "NOUN", "PUNCT"
        };

        String[] casPosCoarse = JCasUtil.select(cas, POS.class)
                .stream()
                .map(POS::getCoarseValue)
                .toArray(String[]::new);

        assertArrayEquals(posCoarse, casPosCoarse);

        String[] deps = new String[]{
                "SB", "--", "NK", "PD", "PG", "NK", "PUNCT", "JU", "SB", "--", "NK", "PD", "PUNCT"
        };

        String[] casDeps = JCasUtil.select(cas, Dependency.class)
                .stream()
                .map(Dependency::getDependencyType)
                .toArray(String[]::new);

        assertArrayEquals(deps, casDeps);

        String[] ents = new String[]{
                // version 3.1.1
                //"MISC", "ORG", "MISC"
                // version 3.2.2
                "ORG"
        };

        String[] casEnts = JCasUtil.select(cas, NamedEntity.class)
                .stream()
                .map(NamedEntity::getValue)
                .toArray(String[]::new);

        assertArrayEquals(ents, casEnts);
    }
}

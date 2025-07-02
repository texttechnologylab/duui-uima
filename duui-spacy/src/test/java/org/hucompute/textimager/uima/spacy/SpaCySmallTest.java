package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.spacy.SpacyToken;
import org.texttechnologylab.uima.type.spacy.SpacyNounChunk;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class SpaCySmallTest {
    static DUUIComposer composer;
    static JCas cas;

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws IOException, SAXException {
        composer.resetPipeline();

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        XmlCasSerializer.serialize(cas.getCas(), null, stream);
        System.out.println(stream.toString(StandardCharsets.UTF_8));

        cas.reset();
    }

    @Test
    public void smallTestEn() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withParameter("use_benepar", String.valueOf(true))
        );

        String language = "en";

        cas.setDocumentText("This is an IPhone by Apple. And this is an iMac.");
        cas.setDocumentLanguage(language);

        composer.run(cas);

        for (SpacyToken token : JCasUtil.select(cas, SpacyToken.class)) {
            System.out.println(token.getCoveredText());
            System.out.println(token.getLemmaValue());
            System.out.println(token.getHasVector());
            float[] vector = token.getVector().toArray();
            List<Float> list = new ArrayList<>();
            for (float value : vector) {
                list.add(value);
            }
            System.out.println(list);
            System.out.println("***");
        }

        for (SpacyNounChunk chunk : JCasUtil.select(cas, SpacyNounChunk.class)) {
            System.out.println(chunk.getCoveredText());
            System.out.println("***");
        }

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
                "DT", "VBZ", "DT", "NN", "IN", "NNP", ".",
                "CC", "DT", "VBZ", "DT", "NNP", "."
        };

        String[] casPos = JCasUtil.select(cas, POS.class)
                .stream()
                .map(POS::getPosValue)
                .toArray(String[]::new);

        assertArrayEquals(pos, casPos);

        String[] posCoarse = new String[]{
                "PRON", "AUX", "DET", "NOUN", "ADP", "PROPN", "PUNCT",
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
                //"ORG"
                // version 3.5
                "ORG", "GPE"
        };

        String[] casEnts = JCasUtil.select(cas, NamedEntity.class)
                .stream()
                .map(NamedEntity::getValue)
                .toArray(String[]::new);

        assertArrayEquals(ents, casEnts);
    }
}

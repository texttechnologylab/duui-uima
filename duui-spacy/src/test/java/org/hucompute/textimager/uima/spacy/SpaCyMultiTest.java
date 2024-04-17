package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.lexmorph.type.pos.POS;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.dependency.Dependency;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XMLSerializer;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import javax.xml.transform.OutputKeys;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class SpaCyMultiTest {
    static final String DOCKER_IMAGE = "docker.texttechnologylab.org/textimager-duui-spacy:0.4.0";

    static DUUIComposer composer;
    static JCas cas;

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

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
    public void emoji1Test() throws Exception {
        composer.add(
            new DUUIDockerDriver.Component(DOCKER_IMAGE)
        );

        String language = "en";

        String text = Files.readString(
                Paths.get("src/test/resources/test1.txt"),
                StandardCharsets.UTF_8
        );

        cas.setDocumentText(text);
        cas.setDocumentLanguage(language);

        composer.run(cas);

        int[][] tokens = new int[][]{
                new int[]{0, 4},
                new int[]{5, 7},
                new int[]{8, 13},
                new int[]{13, 14},
        };

        int[][] casTokens = JCasUtil.select(cas, Token.class)
                .stream()
                .map(s -> new int[] { s.getBegin(), s.getEnd() })
                .toArray(int[][]::new);

        assertArrayEquals(tokens, casTokens);
    }

    @Test
    public void multiTestEn() throws Exception {
        composer.add(
                new DUUIDockerDriver.Component(DOCKER_IMAGE)
        );

        String language = "en";

        cas.setDocumentText("This is an IPhone by Apple. And this is an iMac.");
        cas.setDocumentLanguage(language);

        composer.run(cas);

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

    @Test
    public void multiTestPreEn() throws Exception {
        composer.add(
                new DUUIDockerDriver.Component(DOCKER_IMAGE)
                .withParameter("use_existing_tokens", String.valueOf(true))
                .withParameter("use_existing_sentences", String.valueOf(true))
        );

        String language = "en";

        cas.setDocumentText("This is an IPhone by Apple. And this is an iMac.");
        cas.setDocumentLanguage(language);
        Sentence sentence1 = new Sentence(cas, 0, 27);
        sentence1.addToIndexes();
        Sentence sentence2 = new Sentence(cas, 28, 48);
        sentence2.addToIndexes();


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

        for(int[] token : tokens) {
           Token token1 = new Token(cas, token[0], token[1]);
           token1.addToIndexes();
        }

        composer.run(cas);

        int[][] casTokens = JCasUtil.select(cas, Token.class)
                .stream()
                .map(s -> new int[] { s.getBegin(), s.getEnd() })
                .toArray(int[][]::new);

        JCasUtil.select(cas, Token.class);

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

    @Test
    public void multiTestDe() throws Exception {
        composer.add(
                new DUUIDockerDriver.Component(DOCKER_IMAGE)
        );

        String language = "de";

        cas.setDocumentText("Das ist ein IPhone von Apple. Und das ist ein iMac.");
        cas.setDocumentLanguage(language);

        composer.run(cas);

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
//                "ORG"
                // version 3.4 trf has no NamedEntity parser
                //"ORG"
                // version 3.5
                "MISC", "ORG", "MISC"
        };

        String[] casEnts = JCasUtil.select(cas, NamedEntity.class)
                .stream()
                .map(NamedEntity::getValue)
                .toArray(String[]::new);

        assertArrayEquals(ents, casEnts);
    }
    @Test
    public void biofidTestDe() throws Exception {

//        String file_name = "45808mini";
        String file_name = "tst";
//        String file_name = "45808";
//        String file_name = "27599";
//        String file_name = "sentences_sampled_biodiv";

//        ArrayList<String> texts = new ArrayList<String>();
//        StringBuilder sb = new StringBuilder();

//        try (BufferedReader br = new BufferedReader(new InputStreamReader(
//				new FileInputStream("/mnt/ssd/SRL/example_s.txt"), StandardCharsets.ISO_8859_1))){
//                new FileInputStream("/home/staff_homes/mkonca/tmp/" + file_name + ".txt"), StandardCharsets.UTF_8))) {
//		try (BufferedReader br = new BufferedReader(new FileReader("/mnt/ssd/SRL/example_s.txt"))) {
//            String line = br.readLine();
//            texts.add(line);
//            while (line != null) {
//                sb.append(" ");
//                sb.append(line);
//                line = br.readLine();
//                texts.add(line);
//            }
//        }
        Path path = Paths.get("/home/mkonca/tmp/" + file_name + ".txt");
        String content = Files.readString(path, StandardCharsets.UTF_8);
//        String t = sb.toString();



        System.out.println("************************");
        System.out.println(content.length());
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                ).withSkipVerification(true);

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:8505")//,
        );


//        int iWorkers = 6;
//        composer.add(new DUUIUIMADriver.Component(
//                createEngineDescription(HeidelTimeBioFID.class
//                )).withScale(iWorkers), DUUIUIMADriver.class);
//
        String language = "de";

        JCas jCas = JCasFactory.createJCas();
//        jCas.setDocumentText("Das ist ein IPhone von Apple. Und das ist ein iMac.");
        jCas.setDocumentText(content);

        jCas.setDocumentLanguage(language);

        composer.run(jCas);

        composer.shutdown();

        Path outputXmi = Paths.get("/home/mkonca/tmp/" + file_name + "_spacy.xmi");
        try (OutputStream outputStream = Files.newOutputStream(outputXmi)) {
            XMLSerializer xmlSerializer = new XMLSerializer(outputStream, true);
            xmlSerializer.setOutputProperty(OutputKeys.VERSION, "1.0");
            xmlSerializer.setOutputProperty(OutputKeys.ENCODING, StandardCharsets.UTF_8.toString());
            XmiCasSerializer xmiCasSerializer = new XmiCasSerializer(null);
            xmiCasSerializer.serialize(jCas.getCas(), xmlSerializer.getContentHandler());
        } catch (SAXException e) {
            e.printStackTrace();
        }
    }
}

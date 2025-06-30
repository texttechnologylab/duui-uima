package org.hucompute.textimager.uima.hate;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import org.texttechnologylab.annotation.Hate;
import org.texttechnologylab.annotation.AnnotationComment;

import static org.junit.Assert.assertEquals;

public class MultiTestHate {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
//    static String model = "pol_emo_mDeBERTa";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
//        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
//        composer.addDriver(docker_driver);


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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length()+sentence.length());
            sentenceAnnotation.addToIndexes();
            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void EnglishTest() throws Exception {
//        composer.add(new DUUIDockerDriver.
//                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
//                .withParameter("model_name", model)
//                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                .withScale(1)
//                .withImageFetching());
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );
        List<String> sentences = Arrays.asList(
                "I hate hate it. How can you do that bad thing to me! HOW!",
                "I very happy to be here. I love this place."
        );

        createCas("en", sentences);

        composer.run(cas);

        HashMap<String, String> expected = new HashMap<>();
        expected.put("0_57", "HATE");
        expected.put("58_101", "NonHate");

        Collection<Hate> all_hate = JCasUtil.select(cas, Hate.class);
        for (Hate hate : all_hate) {
            int begin = hate.getBegin();
            int end = hate.getEnd();
            double hate_i = hate.getHate();
            double non_hate = hate.getNonHate();
            String out_i = "HATE";
            if (hate_i < non_hate){
                out_i = "NonHate";
            }
            String expected_i = expected.get(begin+"_"+end);
            Assertions.assertEquals(expected_i, out_i);
        }
    }

    @Test
    public void DeTest() throws Exception {
//        composer.add(new DUUIDockerDriver.
//                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
//                .withParameter("model_name", model)
//                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                .withScale(1)
//                .withImageFetching());
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );
        List<String> sentences = Arrays.asList(
                "I hasse es warum geht es einfach nicht. WARUM!!",
                "I bin glücklich. Dieser Platz ist wunderschön."
        );

        createCas("de", sentences);

        composer.run(cas);

        HashMap<String, String> expected = new HashMap<>();
        expected.put("0_47", "HATE");
        expected.put("48_94", "NonHate");

        Collection<Hate> all_hate = JCasUtil.select(cas, Hate.class);
        for (Hate hate : all_hate) {
            int begin = hate.getBegin();
            int end = hate.getEnd();
            double hate_i = hate.getHate();
            double non_hate = hate.getNonHate();
            String out_i = "HATE";
            if (hate_i < non_hate){
                out_i = "NonHate";
            }
            String expected_i = expected.get(begin+"_"+end);
            Assertions.assertEquals(expected_i, out_i);
        }
    }
}

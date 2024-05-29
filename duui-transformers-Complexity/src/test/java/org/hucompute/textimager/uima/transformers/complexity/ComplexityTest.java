package org.hucompute.textimager.uima.transformers.complexity;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.*;
import org.xml.sax.SAXException;
import org.apache.uima.jcas.tcas.Annotation;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

public class ComplexityTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
//    static String model = "google-bert/bert-base-multilingual-cased";
    static String model = "intfloat/multilingual-e5-base";
//    static String model = "pol_emo_mDeBERTa";
    static String complexities = "euclidean,cosine,wasserstein,distance,jensenshannon,bhattacharyya";

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
    public void EnTest() throws Exception {

//        composer.add(new DUUIDockerDriver.
//                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
//                .withParameter("model_name", model)
//                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                .withScale(1)
//                .withImageFetching());
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("model_art", "Sentence")
                        .withParameter("complexity_compute", complexities)
                        .withParameter("embeddings_keep", "1")
        );

        JCas cas = JCasFactory.createText("The quick brown fox jumps over the lazy dog. I hate this place.");
        cas.setDocumentLanguage("en");


        Annotation a1 = new Annotation(cas, 0, 44);
        a1.addToIndexes();
        Annotation a2 = new Annotation(cas, 45, 63);
        a2.addToIndexes();
        SentenceComparison sc1 = new SentenceComparison(cas);
        sc1.setSentenceI(a1);
        sc1.setSentenceJ(a2);
        sc1.addToIndexes();
        //
        Collection<SentenceComparison> all_sc = JCasUtil.select(cas, SentenceComparison.class);

        System.out.println("Run Composer");

        composer.run(cas);

        composer.shutdown();

        System.out.println("h");

        Collection<Complexity> allcomplex = JCasUtil.select(cas, Complexity.class);
//        assert allcomplex.size() == 36;
        for (Complexity c : allcomplex) {
            String kind = c.getKind();
            double value = c.getOutput();
            switch (kind) {
                case "euclidean", "cosine", "distance" -> {
                    assert value > 0.5;
                }
                case "wasserstein" -> {
                    assert value < 0.1;
                }
                case "jensenshannon", "bhattacharyya" -> {
                    assert value == 0.0;
                }
            }
        }
    }


    @Test
    public void DeTest() throws Exception {

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("model_art", "Sentence")
                        .withParameter("complexity_compute", complexities)
                        .withParameter("embeddings_keep", "1")
        );

        JCas cas = JCasFactory.createText("Der schnelle braune Fuchs springt über den faulen Hund. Ich hasse diesen Ort.");
        cas.setDocumentLanguage("de");


        Annotation a1 = new Annotation(cas, 0, 55);
        a1.addToIndexes();
        Annotation a2 = new Annotation(cas, 56, 77);
        a2.addToIndexes();
        SentenceComparison sc1 = new SentenceComparison(cas);
        sc1.setSentenceI(a1);
        sc1.setSentenceJ(a2);
        sc1.addToIndexes();
        //
        Collection<SentenceComparison> all_sc = JCasUtil.select(cas, SentenceComparison.class);

        System.out.println("Run Composer");

        composer.run(cas);

        composer.shutdown();

        System.out.println("h");

        Collection<Complexity> allcomplex = JCasUtil.select(cas, Complexity.class);
//        assert allcomplex.size() == 36;
        for (Complexity c : allcomplex) {
            String kind = c.getKind();
            double value = c.getOutput();
            switch (kind) {
                case "euclidean", "cosine", "distance" -> {
                    assert value > 0.5;
                }
                case "wasserstein" -> {
                    assert value < 0.1;
                }
                case "jensenshannon", "bhattacharyya" -> {
                    assert value == 0.0;
                }
            }
        }
    }
}

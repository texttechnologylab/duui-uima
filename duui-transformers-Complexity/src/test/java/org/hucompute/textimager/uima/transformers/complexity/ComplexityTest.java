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

public class ComplexityTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
//    static String model = "google-bert/bert-base-multilingual-cased";
    static String model = "paraphrase-multilingual-MiniLM-L12-v2";
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
    public void sentencesTest() throws Exception {

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

        JCas cas = JCasFactory.createText("Justine Bateman is a producer. Justine Tanya Bateman (born February 19, 1966) is an American writer, producer, and actress . She is best known for her regular role as Mallory Keaton on the sitcom Family Ties (1982 -- 1989). Until recently, Bateman ran a production and consulting company, SECTION 5 . In the fall of 2012, she started studying computer science at UCLA. Jane is an engineer. Jane writes code for Huggingface.");
        cas.setDocumentLanguage("en");


        Annotation a1 = new Annotation(cas, 0, 30);
        a1.addToIndexes();
        Annotation a2 = new Annotation(cas, 31, 368);
        a2.addToIndexes();
        Annotation a3 = new Annotation(cas, 369, 389);
        a3.addToIndexes();
        Annotation a4 = new Annotation(cas, 390, 423);
        a4.addToIndexes();
        SentenceComparison sc1 = new SentenceComparison(cas);
        sc1.setSentenceI(a1);
        sc1.setSentenceJ(a2);
        sc1.addToIndexes();
        //
        SentenceComparison sc2 = new SentenceComparison(cas);
        sc2.setSentenceI(a1);
        sc2.setSentenceJ(a3);
        sc2.addToIndexes();

        SentenceComparison sc3 = new SentenceComparison(cas);
        sc3.setSentenceI(a1);
        sc3.setSentenceJ(a4);
        sc3.addToIndexes();

        SentenceComparison sc4 = new SentenceComparison(cas);
        sc4.setSentenceI(a2);
        sc4.setSentenceJ(a3);
        sc4.addToIndexes();

        SentenceComparison sc5 = new SentenceComparison(cas);
        sc5.setSentenceI(a2);
        sc5.setSentenceJ(a4);
        sc5.addToIndexes();

        SentenceComparison sc6 = new SentenceComparison(cas);
        sc6.setSentenceI(a3);
        sc6.setSentenceJ(a4);
        sc6.addToIndexes();
        Collection<SentenceComparison> all_sc = JCasUtil.select(cas, SentenceComparison.class);

        System.out.println("Run Composer");

        composer.run(cas);

        composer.shutdown();

        System.out.println("h");

        Collection<Complexity> allcomplex = JCasUtil.select(cas, Complexity.class);
        assert allcomplex.size() == 36;
//
//        // 1 sentiment per sentence, +1 for average
//        assertEquals(sentences.size()+1, sentiments.size());
//
//        // Sentiments should be neu (average), pos (sentence 1), neg (s2)
//        Double[] expectedSentiments = new Double[]{ 0d, 1d, -1d };
//        Double[] actualSentiments = sentiments.stream().map(Sentiment::getSentiment).toArray(Double[]::new);
//        assertArrayEquals(expectedSentiments, actualSentiments);
    }
}

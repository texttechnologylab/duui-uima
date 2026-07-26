package org.hucompute.textimager.uima.transformers.sentiment;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.testcontainers.shaded.org.checkerframework.checker.units.qual.A;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.*;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SentimentTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
//    static String model = "chkla/parlbert-topic-german";

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
    public void EnCadriffNLPTest() throws Exception {
        HashMap<String, ArrayList<String>> expected1 = new HashMap<>();
        ArrayList<String> expected2 = new ArrayList<>();
        expected2.add("603 - Traditional Morality: Positive");
        expected2.add("305 - Political Authority");
        expected1.put("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1", expected2);
        expected2 = new ArrayList<>();
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                        .withParameter("selection", "text")
        );

        List<String> sentences = Arrays.asList(
                "I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.",
                "These are the latest news from the USA. Joe Biden has won the election."
        );

        createCas("en", sentences);
        composer.run(cas);
        ArrayList<Double> sentiments = new ArrayList<>();
        sentiments.add(0.04603708162903786);
        sentiments.add(0.7044453024864197);
        sentiments.add(0.24951763451099396);
        sentiments.add(0.02839239314198494);
        sentiments.add(0.2581740915775299);
        sentiments.add(0.713433563709259);

        Collection<SentimentModel> all_sentiment = JCasUtil.select(cas, SentimentModel.class);
        String model_name = "";
        Integer counter = 0;
        for (SentimentModel sentiment_i: all_sentiment){
            System.out.println(sentiment_i.getCoveredText());
            Double negative = sentiment_i.getProbabilityNegative();
            Double neutral = sentiment_i.getProbabilityNeutral();
            Double positive = sentiment_i.getProbabilityPositive();
            Integer sentiment = sentiment_i.getSentiment();
            Assertions.assertEquals(negative, sentiments.get(counter));
            Assertions.assertEquals(neutral, sentiments.get(counter+1));
            Assertions.assertEquals(positive, sentiments.get(counter+2));
            counter = counter +3;
            model_name = sentiment_i.getModel().getModelName();
        }
        System.out.printf("Model Name: %s\n", model_name);
    }

    @Test
    public void EnCitizenLabTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.",
                "These are the latest news from the USA. Joe Biden has won the election."
        );

        createCas("en", sentences);
        composer.run(cas);
        ArrayList<Double> sentiments = new ArrayList<>();
        sentiments.add(0.0017020059749484062);
        sentiments.add(0.9876464605331421);
        sentiments.add(0.01065155677497387);
        sentiments.add(0.002740474883466959);
        sentiments.add(0.9765405058860779);
        sentiments.add(0.020719023421406746);

        Collection<SentimentModel> all_sentiment = JCasUtil.select(cas, SentimentModel.class);
        String model_name = "";
        Integer counter = 0;
        for (SentimentModel sentiment_i: all_sentiment){
            model_name = sentiment_i.getModel().getModelName();
            System.out.println(sentiment_i.getCoveredText());
            Double negative = sentiment_i.getProbabilityNegative();
            Double neutral = sentiment_i.getProbabilityNeutral();
            Double positive = sentiment_i.getProbabilityPositive();
//            System.out.println(negative);
//            System.out.println(neutral);
//            System.out.println(positive);
            Assertions.assertEquals(negative, sentiments.get(counter));
            Assertions.assertEquals(neutral, sentiments.get(counter+1));
            Assertions.assertEquals(positive, sentiments.get(counter+2));
            counter = counter +3;
        }
        System.out.printf("Model Name: %s\n", model_name);
    }

    @Test
    public void EnTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.",
                "These are the latest news from the USA. Joe Biden has won the election."
        );

        createCas("en", sentences);
        composer.run(cas);
        Collection<SentimentModel> all_sentiment = JCasUtil.select(cas, SentimentModel.class);
        for (SentimentModel sentiment_i: all_sentiment){
            System.out.println(sentiment_i.getCoveredText());
            Double negative = sentiment_i.getProbabilityNegative();
            Double neutral = sentiment_i.getProbabilityNeutral();
            Double positive = sentiment_i.getProbabilityPositive();
            System.out.println("Negative: " + negative);
            System.out.println("Neutral: " + neutral);
            System.out.println("Positive: " + positive);
        }
    }
}

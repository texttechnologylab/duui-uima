package org.hucompute.textimager.uima.transformers.topic;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;
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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.texttechnologylab.annotation.Topic;
import org.texttechnologylab.annotation.AnnotationComment;

public class TopicTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
//    static String url = "http://tweentopic.service.component.duui.texttechnologylab.org";
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
    public void DeTest() throws Exception {
        HashMap<String, ArrayList<String>> expected1 = new HashMap<>();
        ArrayList<String> expected2 = new ArrayList<>();
        expected2.add("Domestic");
        expected2.add("Technology");
        expected1.put("test", expected2);
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "Ich bin ein Profi-Fußballspieler und spiele bei FC Barcelona in Spanien.",
                "Das sind die Aktuellen Neuigkeiten aus den USA. Joe Biden hat die Wahl gewonnen."
        );

        createCas("de", sentences);
        composer.run(cas);

        Collection<Topic> all_topics = JCasUtil.select(cas, Topic.class);
        ArrayList<Map<String, Float>> expected = new ArrayList<Map<String, Float>>();
        for (Topic topic: all_topics){
            System.out.println(topic.getCoveredText());
            Map<String, Float> topics = new HashMap<String, Float>();
            FSArray<AnnotationComment> topics_all = topic.getTopics();
            for (AnnotationComment comment_i: topics_all){
                topics.put(comment_i.getKey(), Float.parseFloat(comment_i.getValue()));
                System.out.println("key:"+comment_i.getKey()+"; Value:"+comment_i.getValue());
            }
            expected.add(topics);
        }

        for (Map<String, Float> topic: expected){
            // highest value
            String key = Collections.max(topic.entrySet(), Map.Entry.comparingByValue()).getKey();
            Assertions.assertEquals(expected1.get("test").get(expected.indexOf(topic)), key);
        }
    }

    @Test
    public void EnTest() throws Exception {
        HashMap<String, ArrayList<String>> expected1 = new HashMap<>();
        ArrayList<String> expected2 = new ArrayList<>();
        expected2.add("603 - Traditional Morality: Positive");
        expected2.add("305 - Political Authority");
        expected1.put("manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2023-1-1", expected2);
        expected2 = new ArrayList<>();
        expected2.add("Others");
        expected2.add("Others");
        expected1.put("poltextlab/xlm-roberta-large-manifesto-cap", expected2);
        expected2 = new ArrayList<>();
        expected2.add("daily_life");
        expected2.add("pop_culture");
        expected1.put("cardiffnlp/tweet-topic-latest-single", expected2);
        expected2 = new ArrayList<>();
        expected2.add("Instruction");
        expected2.add("News");
        expected1.put("classla/xlm-roberta-base-multilingual-text-genre-classifier", expected2);
        expected2 = new ArrayList<>();
        expected2.add("A7");
        expected2.add("A8");
        expected1.put("ssharoff/genres", expected2);
        expected2 = new ArrayList<>();
        expected2.add("arts, culture, entertainment and media");
        expected2.add("politics");
        expected1.put("classla/multilingual-IPTC-news-topic-classifier", expected2);
        expected2 = new ArrayList<>();
        expected2.add("Culture");
        expected2.add("Government Operations");
        expected1.put("poltextlab/xlm-roberta-large-english-cap-v3", expected2);
        expected2 = new ArrayList<>();
        expected2.add("Others");
        expected2.add("Others");
        expected1.put("poltextlab/xlm-roberta-large-party-cap-v3", expected2);
        expected2 = new ArrayList<>();
        expected2.add("arts_&_culture");
        expected2.add("pop_culture");
        expected1.put("cardiffnlp/roberta-large-tweet-topic-single-all", expected2);
        expected2 = new ArrayList<>();
        expected2.add("other_hobbies");
        expected2.add("news_&_social_concern");
        expected1.put("cardiffnlp/tweet-topic-large-multilingual", expected2);
        expected2 = new ArrayList<>();
        expected2.add("Travel");
        expected2.add("Crime & Law");
        expected1.put("WebOrganizer/TopicClassifier", expected2);
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.",
                "These are the latest news from the USA. Joe Biden has won the election."
        );

        createCas("de", sentences);
        composer.run(cas);

        Collection<Topic> all_topics = JCasUtil.select(cas, Topic.class);
        ArrayList<Map<String, Float>> expected = new ArrayList<Map<String, Float>>();
        for (Topic topic: all_topics){
            System.out.println(topic.getCoveredText());
            Map<String, Float> topics = new HashMap<String, Float>();
            String model_name = topic.getModel().getModelName();
            FSArray<AnnotationComment> topics_all = topic.getTopics();
            for (AnnotationComment comment_i: topics_all){
                topics.put(comment_i.getKey(), Float.parseFloat(comment_i.getValue()));
                System.out.println("key:"+comment_i.getKey()+"; Value:"+comment_i.getValue());
            }
            expected.add(topics);
            String key = Collections.max(topics.entrySet(), Map.Entry.comparingByValue()).getKey();
            Assertions.assertEquals(expected1.get(model_name).get(expected.indexOf(topics)), key);
        }

//        HashMap<String, HashMap<String, Double>> expected = new HashMap<>();
//        Collection<CategoryCoveredTagged> topics = JCasUtil.select(cas, CategoryCoveredTagged.class);
////        System.out.println(topics.size());
//        for (CategoryCoveredTagged topic: topics){
//            int start = topic.getBegin();
//            int end = topic.getEnd();
//            String coveredText = topic.getCoveredText();
//            String value = topic.getValue();
//            double score = topic.getScore();
//            String key1 = start + "_" + end;
//            HashMap<String, Double> value1 = new HashMap<>();
//            value1.put(value, score);
//            if (expected.containsKey(key1)){
//                expected.get(key1).put(value, score);
//            } else {
//                expected.put(key1, value1);
//            }
//        }
//        HashMap<String, String> expected1 = new HashMap<>();
//        expected1.put("0_104", "Instruction");
//        expected1.put("105_176", "News");
//        for (Map.Entry<String, HashMap<String, Double>> entry: expected.entrySet()){
//            String key = Collections.max(entry.getValue().entrySet(), Map.Entry.comparingByValue()).getKey();
//            // compare the expected with same index in the actual
//            String expectedValue = expected1.get(entry.getKey());
//            assertEquals(expectedValue, key);
//        }

    }
}

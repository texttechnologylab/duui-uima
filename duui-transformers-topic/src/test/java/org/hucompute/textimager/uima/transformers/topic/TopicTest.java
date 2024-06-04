package org.hucompute.textimager.uima.transformers.topic;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TopicTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
//    static String model = "chkla/parlbert-topic-german";
    static String model = "classla/xlm-roberta-base-multilingual-text-genre-classifier";

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
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "Ich bin ein Profi-Fußballspieler und spiele bei FC Barcelona in Spanien.",
                "Das sind die Aktuellen Neuigkeiten aus den USA. Joe Biden hat die Wahl gewonnen."
        );

        createCas("de", sentences);
        composer.run(cas);
        HashMap<String, HashMap<String, Double>> expected = new HashMap<>();
        Collection<CategoryCoveredTagged> topics = JCasUtil.select(cas, CategoryCoveredTagged.class);
//        System.out.println(topics.size());
        for (CategoryCoveredTagged topic: topics){
            int start = topic.getBegin();
            int end = topic.getEnd();
            String coveredText = topic.getCoveredText();
            String value = topic.getValue();
            double score = topic.getScore();
            String key1 = start + "_" + end;
            HashMap<String, Double> value1 = new HashMap<>();
            value1.put(value, score);
            if (expected.containsKey(key1)){
                expected.get(key1).put(value, score);
            } else {
                expected.put(key1, value1);
            }
        }
        HashMap<String, String> expected1 = new HashMap<>();
        expected1.put("0_72", "Information/Explanation");
        expected1.put("73_153", "News");
        for (Map.Entry<String, HashMap<String, Double>> entry: expected.entrySet()){
            String key = Collections.max(entry.getValue().entrySet(), Map.Entry.comparingByValue()).getKey();
            // compare the expected with same index in the actual
            String expectedValue = expected1.get(entry.getKey());
            assertEquals(expectedValue, key);
        }
    }

    @Test
    public void EnTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.",
                "These are the latest news from the USA. Joe Biden has won the election."
        );

        createCas("de", sentences);
        composer.run(cas);
        HashMap<String, HashMap<String, Double>> expected = new HashMap<>();
        Collection<CategoryCoveredTagged> topics = JCasUtil.select(cas, CategoryCoveredTagged.class);
//        System.out.println(topics.size());
        for (CategoryCoveredTagged topic: topics){
            int start = topic.getBegin();
            int end = topic.getEnd();
            String coveredText = topic.getCoveredText();
            String value = topic.getValue();
            double score = topic.getScore();
            String key1 = start + "_" + end;
            HashMap<String, Double> value1 = new HashMap<>();
            value1.put(value, score);
            if (expected.containsKey(key1)){
                expected.get(key1).put(value, score);
            } else {
                expected.put(key1, value1);
            }
        }
        HashMap<String, String> expected1 = new HashMap<>();
        expected1.put("0_104", "Instruction");
        expected1.put("105_176", "News");
        for (Map.Entry<String, HashMap<String, Double>> entry: expected.entrySet()){
            String key = Collections.max(entry.getValue().entrySet(), Map.Entry.comparingByValue()).getKey();
            // compare the expected with same index in the actual
            String expectedValue = expected1.get(entry.getKey());
            assertEquals(expectedValue, key);
        }

    }
}

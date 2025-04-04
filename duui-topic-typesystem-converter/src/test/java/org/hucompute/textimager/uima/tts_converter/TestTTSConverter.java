package org.hucompute.textimager.uima.tts_converter;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.TypeSystem;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.CasIOUtils;
import org.apache.uima.util.XmlCasSerializer;

import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.TopicValueBaseWithScore;
import org.texttechnologylab.annotation.UnifiedTopic;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.zip.GZIPInputStream;

public class TestTTSConverter {

    static DUUIComposer composer;
    static JCas cas;
    static String url = "http://127.0.0.1:8000";


    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = (new DUUIComposer()).withSkipVerification(true).withLuaContext((new DUUILuaContext()).withJsonLibrary());
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
        XmlCasSerializer.serialize(cas.getCas(), (TypeSystem)null, stream);
        System.out.println(stream.toString(StandardCharsets.UTF_8));
        cas.reset();
    }

    @Test
    public void TestCategoryCoveredToUnifiedTopic() throws Exception {
        String data_file = "src/test/resources/categorycoveredtag_sentence.xml";

        CasIOUtils.load(
               new FileInputStream(data_file),
                cas.getCas()
        );

        composer.add((new DUUIRemoteDriver.Component(url))
                        .withParameter("remove_old", "true")
                .withParameter("selection", "org.hucompute.textimager.uima.type.category.CategoryCoveredTagged").build().withTimeout(1000L));
        composer.run(cas);

        ArrayList<String> expected1 = new ArrayList<>();
        expected1.add("__label_ddc__700");
        expected1.add("__label_ddc__100");


        Collection<UnifiedTopic> all_topics = JCasUtil.select(cas, UnifiedTopic.class);
        for (UnifiedTopic topic : all_topics) {
            System.out.println(topic.getCoveredText());
            Integer idx = ((ArrayList) all_topics).indexOf(topic);
            Map<String, Double> topics = new HashMap<>();
            FSArray topicArray = topic.getTopics();

            if (topicArray != null) {
                for (int i = 0; i < topicArray.size(); i++) {
                    if (topicArray.get(i) instanceof TopicValueBaseWithScore) {
                        TopicValueBaseWithScore topicValue = (TopicValueBaseWithScore) topicArray.get(i);
                        topics.put(topicValue.getValue(), topicValue.getScore());
                    }
                }
            }

            // Get the key with the highest score
            if (!topics.isEmpty()) {
                String key = Collections.max(topics.entrySet(), Map.Entry.comparingByValue()).getKey();
                System.out.println(key);
                Assertions.assertEquals(expected1.get(idx), key);

            }
        }


    }


    @Test
    public void TestTransformerTopicToUnifiedTopic() throws Exception {
        String data_file = "src/test/resources/transformer_topic.xml";

        CasIOUtils.load(
                new FileInputStream(data_file),
                cas.getCas()
        );

        composer.add((new DUUIRemoteDriver.Component(url))
                .withParameter("remove_old", "true")
                .withParameter("selection", "org.texttechnologylab.annotation.Topic").build().withTimeout(1000L));
        composer.run(cas);

        ArrayList<String> expected1 = new ArrayList<>();
        expected1.add("Domestic");
        expected1.add("Technology");


        Collection<UnifiedTopic> all_topics = JCasUtil.select(cas, UnifiedTopic.class);
        for (UnifiedTopic topic : all_topics) {
            System.out.println(topic.getCoveredText());
            Integer idx = ((ArrayList) all_topics).indexOf(topic);
            Map<String, Double> topics = new HashMap<>();
            FSArray topicArray = topic.getTopics();

            if (topicArray != null) {
                for (int i = 0; i < topicArray.size(); i++) {
                    if (topicArray.get(i) instanceof TopicValueBaseWithScore) {
                        TopicValueBaseWithScore topicValue = (TopicValueBaseWithScore) topicArray.get(i);
                        topics.put(topicValue.getValue(), topicValue.getScore());
                    }
                }
            }

            // Get the key with the highest score
            if (!topics.isEmpty()) {
                String key = Collections.max(topics.entrySet(), Map.Entry.comparingByValue()).getKey();
                System.out.println(key);
                Assertions.assertEquals(expected1.get(idx), key);

            }
        }


    }


    @Test
    public void TestBERTopicToUnifiedTopic() throws Exception {
        String data_file = "src/test/resources/bertopic.xml";

        CasIOUtils.load(
                new FileInputStream(data_file),
                cas.getCas()
        );

        composer.add((new DUUIRemoteDriver.Component(url))
                .withParameter("remove_old", "true")
                .withParameter("selection", "org.texttechnologylab.annotation.BertTopic").build().withTimeout(1000L));
        composer.run(cas);

        ArrayList<String> expected1 = new ArrayList<>();
        expected1.add("1155_biodiversity_extinction_extinctions_ecosystem");
        expected1.add("181_neural_neuron_neurons_convolutions");


        Collection<UnifiedTopic> all_topics = JCasUtil.select(cas, UnifiedTopic.class);
        for (UnifiedTopic topic : all_topics) {
            System.out.println(topic.getCoveredText());
            Integer idx = ((ArrayList) all_topics).indexOf(topic);
            Map<String, Double> topics = new HashMap<>();
            FSArray topicArray = topic.getTopics();

            if (topicArray != null) {
                for (int i = 0; i < topicArray.size(); i++) {
                    if (topicArray.get(i) instanceof TopicValueBaseWithScore) {
                        TopicValueBaseWithScore topicValue = (TopicValueBaseWithScore) topicArray.get(i);
                        topics.put(topicValue.getValue(), topicValue.getScore());
                    }
                }
            }

            // Get the key with the highest score
            if (!topics.isEmpty()) {
                String key = Collections.max(topics.entrySet(), Map.Entry.comparingByValue()).getKey();
                System.out.println(key);
                Assertions.assertEquals(expected1.get(idx), key);

            }
        }


    }
}

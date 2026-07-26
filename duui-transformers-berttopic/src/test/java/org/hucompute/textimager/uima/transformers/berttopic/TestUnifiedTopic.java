package org.hucompute.textimager.uima.transformers.berttopic;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.TypeSystem;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.TopicValueBaseWithScore;
import org.texttechnologylab.annotation.UnifiedTopic;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;


public class TestUnifiedTopic {

    static DUUIComposer composer;
    static JCas cas;
    static String url = "http://127.0.0.1:8000";

    public TestUnifiedTopic() {
    }

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

    public void createCas4P(String language, List<String> paras) throws UIMAException {
        cas.setDocumentLanguage(language);
        StringBuilder sb = new StringBuilder();

        for(String para : paras) {
            Paragraph paraAnnotation = new Paragraph(cas, sb.length(), sb.length() + para.length());
            paraAnnotation.addToIndexes();
            sb.append(para).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void TestBERTopic() throws Exception {
        composer.add((new DUUIRemoteDriver.Component(url)).withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph").build().withTimeout(1000L));
        List<String> paras = Arrays.asList("    Regularization by the sum of singular values, also referred to as the trace norm, is a popular technique for estimating low rank rectangular matrices. In this paper, we extend some of the consistency results of the Lasso to provide necessary and sufficient conditions for rank consistency of trace norm minimization with the square loss. We also provide an adaptive version that is rank consistent even when the necessary condition for the non adaptive version is not fulfilled. ",
                " This paper I assume that in humans the creation of knowledge depends on adiscrete time, or stage, sequential decision-making process subjected to astochastic, information transmitting environment. For each time-stage, thisenvironment randomly transmits Shannon type information-packets to thedecision-maker, who examines each of them for relevancy and then determines hisoptimal choices. Using this set of relevant information-packets, thedecision-maker adapts, over time, to the stochastic nature of his environment,and optimizes the subjective expected rate-of-growth of knowledge. Thedecision-maker's optimal actions, lead to a decision function that involves,over time, his view of the subjective entropy of the environmental process andother important parameters at each time-stage of the process. Using this modelof human behavior, one could create psychometric experiments using computersimulation and real decision-makers, to play programmed games to measure the resulting human performance.");
        this.createCas4P("en", paras);
        composer.run(cas);

        ArrayList<String> expected1 = new ArrayList<>();
        expected1.add("Optimal Transport and Related Methods");
        expected1.add("Remote Sensing and Deep Learning");

        ArrayList<Integer> expected2 = new ArrayList<>();
        expected2.add(27);
        expected2.add(28);

        Collection<UnifiedTopic> all_topics = JCasUtil.select(cas, UnifiedTopic.class);
        for (UnifiedTopic topic : all_topics) {
            System.out.println(topic.getCoveredText());
            Integer idx = ((ArrayList) all_topics).indexOf(topic);
            Map<String, Double> topics = new HashMap<>();
            Map<String, Integer> topic_words_count = new HashMap<>();
            FSArray topicArray = topic.getTopics();

            if (topicArray != null) {
                for (int i = 0; i < topicArray.size(); i++) {
                    if (topicArray.get(i) instanceof TopicValueBaseWithScore) {
                        TopicValueBaseWithScore topicValue = (TopicValueBaseWithScore) topicArray.get(i);
                        Integer num_words = topicValue.getWords().size();

                        topics.put(topicValue.getValue(), topicValue.getScore());
                        topic_words_count.put(topicValue.getValue(), num_words);
                    }
                }
            }

            // Get the key with the highest score
            if (!topics.isEmpty()) {
                String key = Collections.max(topics.entrySet(), Map.Entry.comparingByValue()).getKey();
                System.out.println(key);
                Assertions.assertEquals(expected1.get(idx), key);

                Assertions.assertEquals(expected2.get(idx), topic_words_count.get(key));
            }
        }



    }




}

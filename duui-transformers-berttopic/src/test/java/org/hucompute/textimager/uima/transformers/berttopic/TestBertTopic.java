//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

package org.hucompute.textimager.uima.transformers.berttopic;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.TypeSystem;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.BertTopic;
import org.texttechnologylab.annotation.TopicValue;
import org.xml.sax.SAXException;

public class TestBertTopic {
    static DUUIComposer composer;
    static JCas cas;
    static String url = "http://127.0.0.1:1000";

    public TestBertTopic() {
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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);
        StringBuilder sb = new StringBuilder();

        for(String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length() + sentence.length());
            sentenceAnnotation.addToIndexes();
            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void TestEnSentences() throws Exception {
        composer.add((new DUUIRemoteDriver.Component(url)).withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence").build().withTimeout(1000L));
        List<String> sentences = Arrays.asList("I will guide through the Labyrinth. First you need to find the entrance. Then you need to find the exit.", "These are the latest news from the USA. Joe Biden has won the election.");
        this.createCas("en", sentences);
        composer.run(cas);
        ArrayList<String> expected_topic = new ArrayList();
        expected_topic.add("2374_entrances_subterranean_tunnel_stairs");
        expected_topic.add("9_reelection_election_republican_elections");
        Collection<BertTopic> all_topics = JCasUtil.select(cas, BertTopic.class);

        for(BertTopic topic : all_topics) {
            Integer idx = ((ArrayList)all_topics).indexOf(topic);
            System.out.println(topic.getCoveredText());
            String result_text_1 = topic.getCoveredText();
            Assertions.assertEquals(sentences.get(idx), result_text_1);

            for(TopicValue topic_i : topic.getTopics()) {
                String result_topic = topic_i.getValue();
                System.out.println("Topic:" + result_topic + "; Probability:" + topic_i.getProbability());
                Assertions.assertEquals(expected_topic.get(idx), result_topic);
            }
        }

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
    public void TestEnParagraph() throws Exception {
        composer.add((new DUUIRemoteDriver.Component(url)).withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph").build().withTimeout(1000L));
        List<String> paras = Arrays.asList("Rising temperatures have become the new normal, as greenhouse gas emissions continue to wreak havoc on our planet's ecosystem. From melting glaciers to intensifying storms, the effects of climate change are far-reaching and devastating. Unless we take drastic action to reduce our carbon footprint, we risk irreparably damaging the delicate balance of nature.", "The AI revolution has brought about a new era of innovation, transforming industries from healthcare to finance with its unprecedented capabilities. Machine learning algorithms have enabled us to analyze vast amounts of data, unlocking secrets and solving complex problems that were previously unimaginable. Yet, as we delve deeper into the realm of artificial intelligence, we must also confront the existential risks it poses.");
        this.createCas4P("en", paras);
        composer.run(cas);
        ArrayList<String> expected_topic = new ArrayList();
        expected_topic.add("1155_biodiversity_extinction_extinctions_ecosystem");
        expected_topic.add("181_neural_neuron_neurons_convolutions");
        Collection<BertTopic> all_topics = JCasUtil.select(cas, BertTopic.class);

        for(BertTopic topic : all_topics) {
            Integer idx = ((ArrayList)all_topics).indexOf(topic);
            System.out.println(topic.getCoveredText());
            String result_text_1 = topic.getCoveredText();
            Assertions.assertEquals(paras.get(idx), result_text_1);

            for(TopicValue topic_i : topic.getTopics()) {
                String result_topic = topic_i.getValue();
                System.out.println("Topic:" + result_topic + "; Probability:" + topic_i.getProbability());
                Assertions.assertEquals(expected_topic.get(idx), result_topic);
            }
        }

    }
}

package org.hucompute.textimager.uima.transformers.sentiment;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.hucompute.textimager.uima.type.CategorizedSentiment;
import org.hucompute.textimager.uima.type.Sentiment;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SiebertSentimentRobertaLargeEnglishTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";
    static String model = "siebert/sentiment-roberta-large-english";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

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
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example."
        );

        createCas("en", sentences);
        composer.run(cas);

        Collection<Sentiment> sentiments = JCasUtil.select(cas, Sentiment.class)
                .stream().filter(s -> !(s instanceof CategorizedSentiment)).collect(Collectors.toList());;

        // 1 sentiment per sentence, +1 for average
        assertEquals(sentences.size()+1, sentiments.size());

        // Sentiments should be neu (average), pos (sentence 1), neg (s2)
        Double[] expectedSentiments = new Double[]{ 0d, 1d, -1d };
        Double[] actualSentiments = sentiments.stream().map(Sentiment::getSentiment).toArray(Double[]::new);
        assertArrayEquals(expectedSentiments, actualSentiments);
    }

    @Test
    public void textTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "text")
        );

        List<String> sentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example."
        );

        createCas("en", sentences);
        composer.run(cas);

        Collection<Sentiment> sentiments = JCasUtil.select(cas, Sentiment.class)
                .stream().filter(s -> !(s instanceof CategorizedSentiment)).collect(Collectors.toList());;

        // 1 sentiment for full text
        assertEquals(1, sentiments.size());

        // Sentiment should be negative
        Sentiment sentiment = sentiments.iterator().next();
        assertEquals(-1d, sentiment.getSentiment());
    }

    @Test
    public void emptyTextTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "text")
        );

        List<String> sentences = List.of("");

        createCas("en", sentences);
        composer.run(cas);

        Collection<Sentiment> sentiments = JCasUtil.select(cas, Sentiment.class)
                .stream().filter(s -> !(s instanceof CategorizedSentiment)).collect(Collectors.toList());;

        // 1 sentiment for full text, even if text is empty
        assertEquals(1, sentiments.size());

        // Sentiment should be positive
        Sentiment sentiment = sentiments.iterator().next();
        assertEquals(1d, sentiment.getSentiment());
    }

    @Test
    public void emptySentencesTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = new ArrayList<>();

        createCas("en", sentences);
        composer.run(cas);

        // 0 sentiments, as there is no sentence
        assertEquals(0, JCasUtil.select(cas, Sentiment.class).size());
    }

    @Test
    public void languageTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = new ArrayList<>();

        createCas("UNSUPPORTED", sentences);
        composer.run(cas);

        // 0 sentiments, as the language is not supported
        assertEquals(0, JCasUtil.select(cas, Sentiment.class).size());
    }

    @Test
    public void posTextTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "text")
        );

        List<String> sentences = List.of(
                "Very good night"
        );

        createCas("en", sentences);
        composer.run(cas);

        Collection<Sentiment> sentiments = JCasUtil.select(cas, Sentiment.class)
                .stream().filter(s -> !(s instanceof CategorizedSentiment)).collect(Collectors.toList());;

        // 1 sentiment for full text
        assertEquals(1, sentiments.size());

        // Sentiment should be positive
        Sentiment sentiment = sentiments.iterator().next();
        assertEquals(1d, sentiment.getSentiment());
    }

    @Test
    public void sentencesAndTextTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example."
        );

        createCas("en", sentences);
        composer.run(cas);

        Collection<Sentiment> sentiments = JCasUtil.select(cas, Sentiment.class)
                .stream().filter(s -> !(s instanceof CategorizedSentiment)).collect(Collectors.toList());;

        // 1 sentiment per sentence, +1 for average, +1 for full text
        assertEquals(sentences.size()+1+1, sentiments.size());

        // Sentiments should be neu (text), neu (average sentences), pos (sentence 1), neg (s2)
        Double[] expectedSentiments = new Double[]{ -1d, 0d, 1d, -1d };
        Double[] actualSentiments = sentiments.stream().map(Sentiment::getSentiment).toArray(Double[]::new);
        assertArrayEquals(expectedSentiments, actualSentiments);
    }
}

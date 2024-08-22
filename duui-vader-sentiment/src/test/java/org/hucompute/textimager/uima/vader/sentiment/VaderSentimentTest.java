package org.hucompute.textimager.uima.vader.sentiment;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.hucompute.textimager.uima.type.GerVaderSentiment;
import org.hucompute.textimager.uima.type.Sentiment;
import org.hucompute.textimager.uima.type.VaderSentiment;
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
import java.io.InvalidObjectException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VaderSentimentTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";

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
    public void enTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "This is very great!",
                "I really dislike this.",
                "I hate this car.",
                "I don't dislike the car.",
                "I don't care...",
                "This tool computes the sentiment per sentence."
        );

        createCas("en", sentences);

        composer.run(cas);

        for (Sentiment sentiment : JCasUtil.select(cas, Sentiment.class)) {
            System.out.println(sentiment.getCoveredText() + " -> " + sentiment.getSentiment());
        }

        // 1 -> full text
        // 1 -> sentence average
        // 6 -> per sentences
        assertEquals(6+1+1, JCasUtil.select(cas, VaderSentiment.class).size());
    }

    @Test
    public void frTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "C'est très bien!",
                "Je n'aime vraiment pas ça.",
                "Je déteste cette voiture.",
                "Je ne déteste pas la voiture.",
                "Je m'en fiche...",
                "Cet outil calcule le sentiment par phrase."
        );

        createCas("fr", sentences);

        composer.run(cas);

        for (Sentiment sentiment : JCasUtil.select(cas, Sentiment.class)) {
            System.out.println(sentiment.getCoveredText() + " -> " + sentiment.getSentiment());
        }

        // 1 -> full text
        // 1 -> sentence average
        // 6 -> per sentences
        assertEquals(6+1+1, JCasUtil.select(cas, VaderSentiment.class).size());
    }

    @Test
    public void deTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        List<String> sentences = Arrays.asList(
                "Das gefällt mir durchaus.",
                "Nein, das mag ich nicht.",
                "Ich hasse dieses Auto.",
                "Dieses Auto mag ich nicht.",
                "Ich liebe dieses Auto."
        );

        createCas("de", sentences);

        composer.run(cas);

        for (Sentiment sentiment : JCasUtil.select(cas, Sentiment.class)) {
            System.out.println(sentiment.getCoveredText() + " -> " + sentiment.getSentiment());
        }

        // 1 -> full text
        // 1 -> sentence average
        // 5 -> per sentences
        assertEquals(5+1+1, JCasUtil.select(cas, GerVaderSentiment.class).size());
    }

    @Test
    public void unknownLangTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("selection", "text")
        );

        List<String> sentences = Arrays.asList(
                "Das gefällt mir durchaus.",
                "Nein, das mag ich nicht.",
                "Ich hasse dieses Auto.",
                "Dieses Auto mag ich nicht.",
                "Ich liebe dieses Auto."
        );

        createCas("es", sentences);

        assertThrows(InvalidObjectException.class, () -> composer.run(cas));

        for (Sentiment sentiment : JCasUtil.select(cas, Sentiment.class)) {
            System.out.println(sentiment.getCoveredText() + " -> " + sentiment.getSentiment());
        }
    }
}

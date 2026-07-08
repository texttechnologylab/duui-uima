package org.texttechnology.tools;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.unihd.dbs.uima.types.heideltime.Timex3;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.type.Time;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HeidelTimeExtTest {

    static DUUIComposer composer;
    static JCas cas;
    static String url = getenvOrDefault("HEIDELTIME_EXT_URL", "http://127.0.0.1:9714");

    @BeforeAll
    static void beforeAll() throws Exception {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        cas = JCasFactory.createJCas();
    }

    @AfterEach
    public void afterEach() throws IOException, SAXException {
        composer.resetPipeline();

        if (cas != null) {
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            XmlCasSerializer.serialize(cas.getCas(), null, stream);
            System.out.println(stream.toString(StandardCharsets.UTF_8));
            cas.reset();
        }
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        if (composer != null) {
            composer.shutdown();
        }
    }

    @Test
    public void CommunicationLayerTest() throws Exception {
        Assumptions.assumeTrue(
                serviceAvailable(url),
                "Skipping test because no HeidelTimeExt DUUI service is reachable at " + url
        );

        String communicationLayer = httpGet(url + "/v1/communication_layer");

        assertTrue(communicationLayer.contains("SUPPORTS_SERIALIZE = true"));
        assertTrue(communicationLayer.contains("function serialize"));
        assertTrue(communicationLayer.contains("function deserialize"));
    }

    @Test
    public void HeidelTimeExtRemoteTest() throws Exception {
        Assumptions.assumeTrue(
                serviceAvailable(url),
                "Skipping test because no HeidelTimeExt DUUI service is reachable at " + url
        );

        composer.add(new DUUIRemoteDriver.Component(url));

        createCas(
                "de",
               Arrays.asList(
                        "Am 12. Mai 2024 begann die Exkursion.",
                        "Nach zwei Tagen wurden weitere Proben gesammelt.",
                        "Morgen um 14 Uhr soll ein weiteres Treffen stattfinden."
                )
        );

        composer.run(cas);

        Collection<Timex3> timexAnnotations = JCasUtil.select(cas, Timex3.class);
        Collection<Time> timeAnnotations = JCasUtil.select(cas, Time.class);

        assertFalse(timexAnnotations.isEmpty(), "HeidelTimeExt should create at least one Timex3 annotation");
        assertFalse(timeAnnotations.isEmpty(), "HeidelTimeExt should create at least one TTLab Time annotation");

        for (Timex3 timex3 : timexAnnotations) {
            System.out.println(
                    "TIMEX3: '" + timex3.getCoveredText() + "'" +
                            " begin=" + timex3.getBegin() +
                            " end=" + timex3.getEnd() +
                            " value=" + timex3.getTimexValue() +
                            " type=" + timex3.getTimexType()
            );

            assertTrue(timex3.getBegin() >= 0);
            assertTrue(timex3.getEnd() > timex3.getBegin());
            assertFalse(timex3.getCoveredText().isBlank());
        }

        for (Time time : timeAnnotations) {
            assertTrue(time.getBegin() >= 0);
            assertTrue(time.getEnd() > time.getBegin());
            assertFalse(time.getCoveredText().isBlank());
        }
    }

    private static void createCas(String language, List<String> sentences) {
        cas.setDocumentLanguage(language);

        String text = String.join("\n", sentences);
        cas.setDocumentText(text);

        int offset = 0;
        for (String sentenceText : sentences) {
            int begin = offset;
            int end = begin + sentenceText.length();

            Sentence sentence = new Sentence(cas, begin, end);
            sentence.addToIndexes();

            addWhitespaceTokens(begin, sentenceText);

            offset = end + 1;
        }
    }

    private static void addWhitespaceTokens(int sentenceOffset, String sentenceText) {
        int tokenBegin = -1;

        for (int i = 0; i <= sentenceText.length(); i++) {
            boolean boundary = i == sentenceText.length() || Character.isWhitespace(sentenceText.charAt(i));

            if (!boundary && tokenBegin < 0) {
                tokenBegin = i;
            }

            if (boundary && tokenBegin >= 0) {
                Token token = new Token(cas, sentenceOffset + tokenBegin, sentenceOffset + i);
                token.addToIndexes();
                tokenBegin = -1;
            }
        }
    }

    private static boolean serviceAvailable(String serviceUrl) {
        try {
            URL endpoint = new URL(serviceUrl + "/v1/communication_layer");
            HttpURLConnection connection = (HttpURLConnection) endpoint.openConnection();
            connection.setConnectTimeout(1000);
            connection.setReadTimeout(1000);
            connection.setRequestMethod("GET");
            int responseCode = connection.getResponseCode();
            connection.disconnect();
            return responseCode >= 200 && responseCode < 300;
        } catch (Exception ignored) {
            return false;
        }
    }

    private static String httpGet(String requestUrl) throws IOException {
        URL endpoint = new URL(requestUrl);
        HttpURLConnection connection = (HttpURLConnection) endpoint.openConnection();
        connection.setConnectTimeout(1000);
        connection.setReadTimeout(1000);
        connection.setRequestMethod("GET");

        try (InputStream inputStream = connection.getInputStream()) {
            return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8);
        } finally {
            connection.disconnect();
        }
    }

    private static String getenvOrDefault(String key, String defaultValue) {
        String value = System.getenv(key);

        if (value == null || value.isBlank()) {
            value = System.getProperty(key);
        }

        if (value == null || value.isBlank()) {
            return defaultValue;
        }

        return value;
    }
}

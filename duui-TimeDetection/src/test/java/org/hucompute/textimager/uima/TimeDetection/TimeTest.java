package org.hucompute.textimager.uima.TimeDetection;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.semaf.isotimeml.TimeX3;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.*;

public class TimeTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";

    private static final String DOCUMENT_CREATION_TIME = getenvOrDefault("DOCUMENT_CREATION_TIME", "2026-06-09");

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
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

    @Test
    public void MicrosoftTimeTest() throws Exception {
        runModelTest(
                "microsoft",
                "de",
                germanBioFidSentences(),
                Collections.emptyMap()
        );
    }

    @Test
    public void Tei2GoTimeTest() throws Exception {
        runModelTest(
                "tei2go",
                "de",
                germanBioFidSentences(),
                Collections.emptyMap()
        );
    }

    @Test
    public void TimexyTimeTest() throws Exception {
        runModelTest(
                "timexy",
                "de",
                germanBioFidSentences(),
                Collections.emptyMap()
        );
    }

    @Test
    public void GermanGelectraTimeTest() throws Exception {
        runModelTest(
                "german-gelectra",
                "de",
                germanBioFidSentences(),
                Collections.emptyMap()
        );
    }

    @Test
    public void BertGotADateTimeTest() throws Exception {
        runModelTest(
                "bert-got-a-date",
                "en",
                englishSentences(),
                Collections.emptyMap()
        );
    }

    @Test
    public void DucklingTimeTest() throws Exception {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("duckling_url", getenvOrDefault("DUCKLING_URL", "http://127.0.0.1:8000"));
        parameters.put("duckling_timezone", getenvOrDefault("DUCKLING_TIMEZONE", "Europe/Berlin"));

        runModelTest(
                "duckling",
                "de",
                germanBioFidSentences(),
                parameters
        );
    }

    @Test
    public void SutimeTimeTest() throws Exception {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("corenlp_url", getenvOrDefault("CORENLP_URL", "http://127.0.0.1:9000"));

        runModelTest(
                "sutime",
                "de",
                germanBioFidSentences(),
                parameters
        );
    }

    @Test
    public void GenericHfTokenClassificationTimeTest() throws Exception {
        runModelTest(
                "hf-token-classification",
                getenvOrDefault("HF_TOKEN_CLASSIFICATION_LANG", "de"),
                germanBioFidSentences(),
                Collections.emptyMap()
        );
    }

    private void runModelTest(
            String expectedModel,
            String language,
            List<String> sentences,
            Map<String, String> extraParameters
    ) throws Exception {
        Assumptions.assumeTrue(
                serviceAvailable(url),
                "Skipping " + expectedModel + " because no DUUI Time service is reachable at " + url
        );

        String runningModel = getRunningModel();
        Assumptions.assumeTrue(
                expectedModel.equals(runningModel),
                "Skipping " + expectedModel + " because running model is " + runningModel + " at " + url
        );

        composer.add(createComponent(extraParameters));

        createCasFromSentences(language, sentences);

        System.out.println("============================================================");
        System.out.println("Expected model: " + expectedModel);
        System.out.println("Running model: " + runningModel);
        System.out.println("URL: " + url);
        System.out.println("Language: " + language);
        System.out.println("Input document:");
        System.out.println(cas.getDocumentText());

        composer.run(cas);

        Collection<TimeX3> timeAnnotations = JCasUtil.select(cas, TimeX3.class);
        Map<String, ArrayList<Object>> result = extractTimeResult();

        for (TimeX3 timeAnnotation : timeAnnotations) {
            String coveredText = timeAnnotation.getCoveredText();
            int begin = timeAnnotation.getBegin();
            int end = timeAnnotation.getEnd();
            String value = timeAnnotation.getValue();
            String typeName = timeAnnotation.getType().getName();

            result.get("token").add(coveredText);
            result.get("begin").add(begin);
            result.get("end").add(end);
            result.get("value").add(value);
            result.get("type").add(typeName);

            System.out.println(
                    "TIMEX3 [" + runningModel + "]: '" + coveredText + "'"
                            + " (begin=" + begin + ", end=" + end + ")"
                            + " value='" + value + "'"
                            + " type='" + typeName + "'"
            );

            assertTrue(begin >= 0, "TimeX3 begin offset must be non-negative");
            assertTrue(end > begin, "TimeX3 end offset must be greater than begin offset");
            assertFalse(coveredText.isBlank(), "TimeX3 covered text must not be blank");

            assertTrue(
                    typeName.equals("org.texttechnologylab.annotation.semaf.isotimeml.TimeX3")
                            || typeName.equals("org.texttechnologylab.annotation.semaf.isotimeml.time.Date")
                            || typeName.equals("org.texttechnologylab.annotation.semaf.isotimeml.time.Time")
                            || typeName.equals("org.texttechnologylab.annotation.semaf.isotimeml.time.Duration")
                            || typeName.equals("org.texttechnologylab.annotation.semaf.isotimeml.time.Set"),
                    "Unexpected TimeX3 type: " + typeName
            );
        }

        assertFalse(
                timeAnnotations.isEmpty(),
                "The DUUI TimeX3 component should create at least one TimeX3 annotation for model " + runningModel
        );
    }

    private DUUIRemoteDriver.Component createComponent(Map<String, String> extraParameters) throws URISyntaxException, IOException {
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(url)
                .withParameter(
                        "selection",
                        "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
                )
                .withParameter("document_creation_time", DOCUMENT_CREATION_TIME)
                .withParameter("threshold", getenvOrDefault("TIME_THRESHOLD", "0.0"))
                .withParameter("batch_size", getenvOrDefault("TIME_BATCH_SIZE", "8"));

        for (Map.Entry<String, String> parameter : extraParameters.entrySet()) {
            if (parameter.getValue() != null && !parameter.getValue().isBlank()) {
                component = component.withParameter(parameter.getKey(), parameter.getValue());
            }
        }

        return component;
    }

    public void createCasFromSentences(String language, List<String> sentences) {
        cas.setDocumentLanguage(language);

        StringBuilder documentText = new StringBuilder();
        int offset = 0;

        for (int i = 0; i < sentences.size(); i++) {
            String sentenceText = sentences.get(i);

            if (i > 0) {
                documentText.append("\n");
                offset += 1;
            }

            int begin = offset;
            int end = begin + sentenceText.length();

            documentText.append(sentenceText);

            Sentence sentence = new Sentence(cas, begin, end);
            sentence.addToIndexes();

            offset = end;
        }

        cas.setDocumentText(documentText.toString());
    }

    private Map<String, ArrayList<Object>> extractTimeResult() {
        Map<String, ArrayList<Object>> result = new LinkedHashMap<>();

        result.put("begin", new ArrayList<>());
        result.put("end", new ArrayList<>());
        result.put("token", new ArrayList<>());
        result.put("value", new ArrayList<>());
        result.put("type", new ArrayList<>());

        return result;
    }

    private static List<String> germanBioFidSentences() {
        return Arrays.asList(
                "Dr. Anna Weber begann die Exkursion am 12. Mai 2024 um 08:30 Uhr in Frankfurt am Main.",
                "Bereits am frühen Morgen wurden die ersten Proben genommen.",
                "Nach zwei Stunden erreichte das Team eine Streuobstwiese am Rand des Taunus.",
                "Dort wurden zwischen 10:15 Uhr und 11:45 Uhr mehrere Insekten beobachtet.",
                "Am Nachmittag dokumentierte BioFID weitere Funde im Labor.",
                "Die zweite Untersuchung fand am folgenden Montag statt.",
                "Jeden Dienstag wurden die Temperaturwerte erneut kontrolliert.",
                "Im Sommer 2024 verglich das Team die Daten mit älteren Beobachtungen.",
                "Vom 1. Juni bis zum 3. Juni wurden zusätzliche Bodenproben gesammelt.",
                "Die Auswertung dauerte drei Wochen und wurde gestern abgeschlossen.",
                "Morgen um 14 Uhr soll ein weiteres Treffen stattfinden.",
                "In zwei Monaten sollen die Ergebnisse erneut überprüft werden.",
                "Seit 2021 werden die Funde regelmäßig in einer Datenbank gespeichert.",
                "Vor drei Jahren begann das Projekt mit einer Pilotstudie am Bodensee."
        );
    }

    private static List<String> englishSentences() {
        return Arrays.asList(
                "Dr. Anna Weber started the field trip on May 12, 2024 at 08:30 in Frankfurt am Main.",
                "After two hours the team reached an orchard near the Taunus.",
                "Several insects were observed between 10:15 and 11:45.",
                "The second investigation took place the following Monday.",
                "Every Tuesday the temperature values were checked again.",
                "The evaluation lasted three weeks and was completed yesterday.",
                "Tomorrow at 2 pm another meeting is scheduled.",
                "In two months the results will be reviewed again.",
                "Since 2021 the findings have been stored regularly in a database.",
                "Three years ago the project began with a pilot study at Lake Constance."
        );
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

    private static String getRunningModel() {
        try {
            String documentation = httpGet(url + "/v1/documentation");

            Pattern selectedModelPattern = Pattern.compile("\"selected_model\"\\s*:\\s*\"([^\"]+)\"");
            Matcher selectedModelMatcher = selectedModelPattern.matcher(documentation);
            if (selectedModelMatcher.find()) {
                return selectedModelMatcher.group(1);
            }

            Pattern modelNamePattern = Pattern.compile("\"model_name\"\\s*:\\s*\"([^\"]+)\"");
            Matcher modelNameMatcher = modelNamePattern.matcher(documentation);
            if (modelNameMatcher.find()) {
                return modelNameMatcher.group(1);
            }

            return "unknown";
        } catch (Exception ex) {
            return "unknown";
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
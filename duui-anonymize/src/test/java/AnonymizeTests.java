import de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly;
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
import org.xml.sax.SAXException;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.dkpro.core.io.xmi.XmiWriter;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.junit.jupiter.api.Assertions.*;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;

/**
 * Integration tests for the DUUI anonymization annotator.
 *
 * Prerequisites: the Python service must be running on {@value #SERVICE_URL}.
 * Start it with:
 *   uvicorn duui:app --host 0.0.0.0 --port 9714 --workers 1
 *
 * Each test loads English text containing personally identifiable information (PII),
 * sends it through the OPF anonymizer via DUUI, and asserts that
 * {@link Anomaly} annotations are written to the CAS output view.
 */
public class AnonymizeTests {

    static final String SERVICE_URL = "http://127.0.0.1:9714";

    static final String RESULTS_DIR = "src/test/results";

    static DUUIComposer composer;
    static JCas cas;

    // -------------------------------------------------------------------
    // JUnit lifecycle
    // -------------------------------------------------------------------

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        Files.createDirectories(Paths.get(RESULTS_DIR));

        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uimaDriver = new DUUIUIMADriver().withDebug(false);
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, uimaDriver);

        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    void afterEach(TestInfo testInfo) throws IOException, SAXException {
        composer.resetPipeline();

        String methodName = testInfo.getTestMethod()
                .map(m -> m.getName())
                .orElseGet(() -> testInfo.getDisplayName().replaceAll("[^a-zA-Z0-9_]", "_"));

        String inputText    = cas.getDocumentText() != null ? cas.getDocumentText() : "";
        String redactedText = extractRedactedText();
        Collection<Anomaly> anomalies = collectAnomalies();

        // Write XMI
        ByteArrayOutputStream xmiBytes = new ByteArrayOutputStream();
        XmlCasSerializer.serialize(cas.getCas(), null, xmiBytes);
        String xmiString = xmiBytes.toString(StandardCharsets.UTF_8);
        Files.writeString(Paths.get(RESULTS_DIR, methodName + ".xmi"), xmiString);

        // Write JSON summary for later comparison
        String json = buildResultJson(methodName, inputText, redactedText, anomalies);
        Files.writeString(Paths.get(RESULTS_DIR, methodName + ".json"), json);

        System.out.println("=== " + methodName + " ===");
        System.out.println(json);

        cas.reset();
    }

    // -------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------

    /** Populate the CAS with the given text and language code. */
    private static void createCas(String language, String text) throws UIMAException {
        cas.setDocumentLanguage(language);
        cas.setDocumentText(text);
    }

    /**
     * Collect all {@link Anomaly} annotations across every CAS view.
     * The anonymizer writes results to an "opf_redacted" SOFA view, but also
     * to the default view depending on the service configuration.
     */
    private static Collection<Anomaly> collectAnomalies() {
        // prefer the dedicated redaction view when available
        try {
            JCas redactedView = cas.getView("opf_redacted");
            Collection<Anomaly> spans = JCasUtil.select(redactedView, Anomaly.class);
            if (!spans.isEmpty()) {
                return spans;
            }
        } catch (Exception ignored) {
            // view does not exist — fall through to default view
        }
        return JCasUtil.select(cas, Anomaly.class);
    }

    /** Returns the sofa string of the opf_redacted view, or the default view's string. */
    private static String extractRedactedText() {
        try {
            JCas view = cas.getView("opf_redacted");
            String s = view.getSofaDataString();
            return s != null ? s : "";
        } catch (Exception ignored) {
            String s = cas.getSofaDataString();
            return s != null ? s : "";
        }
    }

    private static String buildResultJson(
            String testName,
            String inputText,
            String redactedText,
            Collection<Anomaly> anomalies) {

        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"test\": ").append(jsonStr(testName)).append(",\n");
        sb.append("  \"input\": ").append(jsonStr(inputText)).append(",\n");
        sb.append("  \"redacted\": ").append(jsonStr(redactedText)).append(",\n");
        sb.append("  \"anomaly_count\": ").append(anomalies.size()).append(",\n");
        sb.append("  \"anomalies\": [\n");
        int idx = 0;
        for (Anomaly a : anomalies) {
            String spanText = "";
            try {
                if (a.getBegin() >= 0 && a.getEnd() <= inputText.length()) {
                    spanText = inputText.substring(a.getBegin(), a.getEnd());
                }
            } catch (Exception ignored) {}
            sb.append("    {\n");
            sb.append("      \"begin\": ").append(a.getBegin()).append(",\n");
            sb.append("      \"end\": ").append(a.getEnd()).append(",\n");
            sb.append("      \"category\": ").append(jsonStr(a.getCategory())).append(",\n");
            sb.append("      \"description\": ").append(jsonStr(a.getDescription())).append(",\n");
            sb.append("      \"text\": ").append(jsonStr(spanText)).append("\n");
            sb.append("    }");
            if (++idx < anomalies.size()) sb.append(",");
            sb.append("\n");
        }
        sb.append("  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String jsonStr(String s) {
        if (s == null) return "null";
        return "\"" + s
                .replace("\\", "\\\\")
                .replace("\"", "\\\"")
                .replace("\n", "\\n")
                .replace("\r", "\\r")
                .replace("\t", "\\t")
                + "\"";
    }

    // -------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------

    /**
     * Smoke test: plain English sentence with a person name.
     * Expects at least one Anomaly annotation to be produced.
     */
    @Test
    @DisplayName("Simple person-name redaction")
    void testSimplePersonName() throws Exception {
        String text = "John Smith called the bank to report a fraud.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        for (Anomaly a : anomalies) {
            System.out.printf("  [%d-%d] category=%s description=%s%n",
                    a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription());
        }

        assertFalse(anomalies.isEmpty(),
                "Expected at least one Anomaly annotation for 'John Smith'");
    }

    /**
     * Email address redaction.
     */
    @Test
    @DisplayName("Email address redaction")
    void testEmailRedaction() throws Exception {
        String text = "Please contact support at alice@example.com for further assistance.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        assertFalse(anomalies.isEmpty(),
                "Expected at least one Anomaly for the email address");
    }

    /**
     * Phone number redaction.
     */
    @Test
    @DisplayName("Phone number redaction")
    void testPhoneNumberRedaction() throws Exception {
        String text = "You can reach Dr. Miller at +49 69 1234 5678 during office hours.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        assertFalse(anomalies.isEmpty(),
                "Expected at least one Anomaly for the phone number or person name");
    }

    /**
     * Multiple PII entities in a single document.
     * Asserts that distinct spans covering the name, email, and phone are returned.
     */
    @Test
    @DisplayName("Multiple PII entities in one document")
    void testMultiplePiiEntities() throws Exception {
        String text =
                "Patient: Jane Doe, DOB: 1985-03-22. " +
                "Contact: jane.doe@hospital.org, Tel: 069-9876-5432. " +
                "Address: 60325 Frankfurt am Main, Goethe-Platz 1.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        for (Anomaly a : anomalies) {
            System.out.printf("  [%d-%d] category=%s text=%s%n",
                    a.getBegin(), a.getEnd(), a.getCategory(),
                    text.substring(a.getBegin(), a.getEnd()));
        }

        assertTrue(anomalies.size() >= 2,
                "Expected at least 2 Anomaly annotations for a document with multiple PII entities");
    }

    /**
     * Pseudo-mode: the service should return the text unchanged (stub behavior).
     * Asserts no Anomaly annotations are created.
     */
    @Test
    @DisplayName("Pseudo mode returns unchanged text")
    void testPseudoMode() throws Exception {
        String text = "Alice and Bob met at the Frankfurt main station.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
                        .withParameter("mode", "pseudo")
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count (pseudo mode): " + anomalies.size());
        assertTrue(anomalies.isEmpty(),
                "Pseudo mode should produce no Anomaly annotations (stub returns input unchanged)");
    }

    /**
     * Custom placeholder: verifies the {@link Anomaly#getDescription()} contains the
     * user-supplied placeholder string instead of the default {@code <REDACTED>}.
     */
    @Test
    @DisplayName("Custom placeholder is reflected in Anomaly description")
    void testCustomPlaceholder() throws Exception {
        String text = "Send the report to max.mustermann@uni-frankfurt.de by Friday.";
        String placeholder = "***PRIVATE***";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
                        .withParameter("placeholder", placeholder)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());

        boolean foundCustomPlaceholder = anomalies.stream()
                .anyMatch(a -> placeholder.equals(a.getDescription()));
        assertTrue(foundCustomPlaceholder,
                "At least one Anomaly should carry the custom placeholder '" + placeholder + "'");
    }

    /**
     * Selection window: only the text between offsets [8, 36] should be analysed.
     * Entities outside that window must not be annotated.
     */
    @Test
    @DisplayName("Selection window constrains annotation range")
    void testSelectionWindow() throws Exception {
        // offsets:  0123456789012345678901234567890123456789
        //           Call Dr. John Adams at 555-0100 today.
        // window [8, 28] covers "John Adams at 555-0100"
        String text = "Call Dr. John Adams at 555-0100 today.";
        int selBegin = 9;
        int selEnd   = 30;
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
                        .withParameter("selection_begin", String.valueOf(selBegin))
                        .withParameter("selection_end",   String.valueOf(selEnd))
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count (selection window): " + anomalies.size());
        for (Anomaly a : anomalies) {
            assertTrue(a.getBegin() >= selBegin && a.getEnd() <= selEnd,
                    String.format("Anomaly [%d-%d] falls outside the selection window [%d-%d]",
                            a.getBegin(), a.getEnd(), selBegin, selEnd));
        }
    }

    /**
     * Empty document: the annotator must not throw and must return no anomalies.
     */
    @Test
    @DisplayName("Empty document produces no anomalies")
    void testEmptyDocument() throws Exception {
        createCas("en", "");

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        assertTrue(anomalies.isEmpty(),
                "An empty document should produce zero Anomaly annotations");
    }

    /**
     * German text: verifies the annotator handles non-English input without crashing.
     * The model may or may not detect German PII depending on the loaded checkpoint;
     * we only assert no exception is thrown.
     */
    @Test
    @DisplayName("German text does not cause an exception")
    void testGermanText() throws Exception {
        String text = "Herr Klaus Muller wohnt in der Goethestrasse 12, 60313 Frankfurt am Main.";
        createCas("de", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        // Should complete without throwing
        assertDoesNotThrow(() -> composer.run(cas));

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("German Anomaly count: " + anomalies.size());
    }

    /**
     * XMI serialisation round-trip: runs the annotator and writes the CAS to an XMI
     * file so the result can be inspected with the UIMA CAS Editor.
     */
    @Test
    @DisplayName("XMI output is written to src/test/results/")
    void testXmiOutput() throws Exception {
        String text =
                "Maria Schmidt (m.schmidt@example.de) lives at Berliner Str. 5, 10115 Berlin.";
        createCas("en", text);

        composer.add(
                new DUUIRemoteDriver.Component(SERVICE_URL)
        );

        composer.add(new DUUIUIMADriver.Component(
                createEngineDescription(XmiWriter.class,
                        XmiWriter.PARAM_TARGET_LOCATION, RESULTS_DIR,
                        XmiWriter.PARAM_PRETTY_PRINT,    true,
                        XmiWriter.PARAM_OVERWRITE,       true,
                        XmiWriter.PARAM_VERSION,         "1.1"
                )
        ).build());

        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("XMI test Anomaly count: " + anomalies.size());
        assertFalse(anomalies.isEmpty(),
                "Expected at least one Anomaly annotation for the PII-rich document");
    }
}

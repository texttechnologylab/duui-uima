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
     * Collect all {@link Anomaly} annotations from the default CAS view.
     * Anomalies are always indexed against the original document text so that
     * their character offsets are valid. The "opf_redacted" view only carries
     * the redacted sofa string and no annotations.
     */
    private static Collection<Anomaly> collectAnomalies() {
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
        sb.append("  \"output\": ").append(jsonStr(redactedText)).append(",\n");
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
    // Mode tests
    // -------------------------------------------------------------------

    /** Placeholder mode (default): PII replaced with [category] tag in redacted_text and Anomaly description. */
    @Test
    @DisplayName("Placeholder mode: PII replaced with [category] tag")
    void testPlaceholderMode() throws Exception {
        String text = "Send the report to max.mustermann@uni-frankfurt.de by Friday.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        anomalies.forEach(a -> System.out.printf("  [%d-%d] category=%s description=%s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(), "Expected at least one Anomaly for the email address");
        assertTrue(anomalies.stream().anyMatch(a -> {
            String d = a.getDescription();
            return d != null && d.startsWith("[") && d.endsWith("]");
        }), "Anomaly description should be a bracketed [category] tag in placeholder mode");

        String redacted = extractRedactedText();
        assertFalse(redacted.contains("max.mustermann@uni-frankfurt.de"),
                "Redacted text should not contain the original email");
        assertTrue(redacted.contains("[private_email]") || redacted.contains("[private_person]"),
                "Redacted text should contain a [category] replacement tag");
    }

    /** Remove mode: PII spans are deleted from redacted_text; Anomaly description is the original word. */
    @Test
    @DisplayName("Remove mode: PII deleted from redacted text")
    void testRemoveMode() throws Exception {
        String text = "Call John Smith at john.smith@company.com or +1-800-555-0199 for help.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "remove"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        anomalies.forEach(a -> System.out.printf("  [%d-%d] category=%s text=%s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), text.substring(a.getBegin(), a.getEnd())));

        assertFalse(anomalies.isEmpty(), "Expected anomalies in remove mode");

        String redacted = extractRedactedText();
        System.out.printf("  original (%d): %s%n", text.length(), text);
        System.out.printf("  redacted (%d): %s%n", redacted.length(), redacted);
        assertTrue(redacted.length() < text.length(),
                "Redacted text should be shorter after PII removal");
        // original PII tokens must be absent from the redacted string
        for (Anomaly a : anomalies) {
            String pii = text.substring(a.getBegin(), a.getEnd());
            assertFalse(redacted.contains(pii),
                    "Removed PII token '" + pii + "' should not appear in redacted text");
        }
    }

    /** Pseudo mode: not yet supported - service returns input unchanged with no annotations. */
    @Test
    @DisplayName("Pseudo mode: not yet supported, returns input unchanged")
    void testPseudoMode() throws Exception {
        String text = "Alice and Bob met at the Frankfurt main station.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "pseudo"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count (pseudo mode): " + anomalies.size());
        assertTrue(anomalies.isEmpty(),
                "Pseudo mode (unsupported stub) should produce no Anomaly annotations");
    }

    // -------------------------------------------------------------------
    // PII type tests  (mode=placeholder so description = [category])
    // -------------------------------------------------------------------

    /** private_person: full name in a simple sentence. */
    @Test
    @DisplayName("Type: private_person")
    void testTypePerson() throws Exception {
        String text = "John Smith called the bank to report a fraud.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(), "Expected at least one annotation");
        assertTrue(anomalies.stream().anyMatch(a -> "private_person".equals(a.getCategory())),
                "Expected category 'private_person' for 'John Smith'");
    }

    /** private_email: plain email address. */
    @Test
    @DisplayName("Type: private_email")
    void testTypeEmail() throws Exception {
        String text = "Please contact alice@example.com for further assistance.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertTrue(anomalies.stream().anyMatch(a -> "private_email".equals(a.getCategory())),
                "Expected category 'private_email' for 'alice@example.com'");
    }

    /** private_phone: international phone number. */
    @Test
    @DisplayName("Type: private_phone")
    void testTypePhone() throws Exception {
        String text = "You can reach Dr. Miller at +49 69 1234 5678 during office hours.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(), "Expected phone or person annotation");
        long phoneCount = anomalies.stream().filter(a -> "private_phone".equals(a.getCategory())).count();
        System.out.println("private_phone spans: " + phoneCount);
        assertTrue(phoneCount > 0, "Expected category 'private_phone' for '+49 69 1234 5678'");
    }

    /** private_address: street address with postcode. */
    @Test
    @DisplayName("Type: private_address")
    void testTypeAddress() throws Exception {
        String text = "She lives at 742 Evergreen Terrace, Springfield, IL 62704.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        long addrCount = anomalies.stream().filter(a -> "private_address".equals(a.getCategory())).count();
        System.out.println("private_address spans: " + addrCount);
        assertTrue(addrCount > 0, "Expected category 'private_address' for the street address");
    }

    /** private_url: personal homepage URL. */
    @Test
    @DisplayName("Type: private_url")
    void testTypeUrl() throws Exception {
        String text = "My personal page is at https://janedoe.personal-site.com/about and I post there.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        long urlCount = anomalies.stream().filter(a -> "private_url".equals(a.getCategory())).count();
        System.out.println("private_url spans: " + urlCount);
        assertTrue(urlCount > 0, "Expected category 'private_url' for the personal URL");
    }

    /** private_date: personally identifying date (e.g. birth date). */
    @Test
    @DisplayName("Type: private_date")
    void testTypeDate() throws Exception {
        String text = "Jane Doe was born on March 15, 1990 in Chicago.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(), "Expected at least one annotation (person or date)");
        long dateCount = anomalies.stream().filter(a -> "private_date".equals(a.getCategory())).count();
        System.out.println("private_date spans: " + dateCount);
        assertTrue(dateCount > 0, "Expected category 'private_date' for 'March 15, 1990'");
    }

    /** account_number: credit-card style number string. */
    @Test
    @DisplayName("Type: account_number")
    void testTypeAccountNumber() throws Exception {
        String text = "Please transfer funds to account number 4532-0151-1283-0366 at Deutsche Bank.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        long acctCount = anomalies.stream().filter(a -> "account_number".equals(a.getCategory())).count();
        System.out.println("account_number spans: " + acctCount);
        assertTrue(acctCount > 0, "Expected category 'account_number' for the card number");
    }

    /** secret: API key / credential in text. */
    @Test
    @DisplayName("Type: secret")
    void testTypeSecret() throws Exception {
        String text = "The API key is sk-proj-abc123XYZ987 and the password is H@nt3r2secure!.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        long secretCount = anomalies.stream().filter(a -> "secret".equals(a.getCategory())).count();
        System.out.println("secret spans: " + secretCount);
        assertTrue(secretCount > 0, "Expected category 'secret' for API key / password");
    }

    // -------------------------------------------------------------------
    // Feature / combination tests
    // -------------------------------------------------------------------

    /** Multiple PII types in one document; verifies distinct categories are detected. */
    @Test
    @DisplayName("Multiple PII types in one document")
    void testMultiplePiiEntities() throws Exception {
        String text =
                "Patient: Jane Doe, DOB: 1985-03-22. " +
                "Contact: jane.doe@hospital.org, Tel: 069-9876-5432. " +
                "Address: 60325 Frankfurt am Main, Goethe-Platz 1.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = '%s'%n",
                a.getBegin(), a.getEnd(), a.getCategory(), text.substring(a.getBegin(), a.getEnd())));

        assertTrue(anomalies.size() >= 2,
                "Expected at least 2 distinct PII annotations");

        long distinctCategories = anomalies.stream().map(Anomaly::getCategory).distinct().count();
        System.out.println("Distinct categories: " + distinctCategories);
        assertTrue(distinctCategories >= 2,
                "Expected annotations from at least 2 different PII categories");
    }

    /** Smoke test with two PII types in one sentence. */
    @Test
    @DisplayName("Smoke test: person + email in one sentence")
    void testSimplePerson() throws Exception {
        String text = "My name is Harry Potter and my email is harry.potter@hogwarts.edu.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(),
                "Expected at least one Anomaly annotation");
    }

    /** Ambiguous context where person identity is inferred from surrounding detail. */
    @Test
    @DisplayName("Complex context: identity inferred from description")
    void testComplexContext() throws Exception {
        String text = "His name is Harry, he works at the TTLAB in Frankfurt, " +
                      "he's the only Chinese guy in the office.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "remove")); // or remove/placeholder mode, should still detect the same spans 
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count: " + anomalies.size());
        anomalies.forEach(a -> System.out.printf("  [%d-%d] %s = %s%n",
                a.getBegin(), a.getEnd(), a.getCategory(), a.getDescription()));

        assertFalse(anomalies.isEmpty(), "Expected at least one annotation in complex context");
    }

    /** Selection window: only span offsets within [selBegin, selEnd] must be annotated. */
    @Test
    @DisplayName("Selection window constrains annotation range")
    void testSelectionWindow() throws Exception {
        // window [9, 30] covers "John Adams at 555-0100"
        String text = "Call Dr. John Adams at 555-0100 today.";
        int selBegin = 9;
        int selEnd   = 30;
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder")
                .withParameter("selection_begin", String.valueOf(selBegin))
                .withParameter("selection_end",   String.valueOf(selEnd)));
        composer.run(cas);

        Collection<Anomaly> anomalies = collectAnomalies();
        System.out.println("Anomaly count (selection window): " + anomalies.size());
        for (Anomaly a : anomalies) {
            assertTrue(a.getBegin() >= selBegin && a.getEnd() <= selEnd,
                    String.format("Anomaly [%d-%d] outside window [%d-%d]",
                            a.getBegin(), a.getEnd(), selBegin, selEnd));
        }
    }

    /** Empty document must not throw and must return zero annotations. */
    @Test
    @DisplayName("Empty document produces no anomalies")
    void testEmptyDocument() throws Exception {
        createCas("en", "");

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));
        composer.run(cas);

        assertTrue(collectAnomalies().isEmpty(),
                "An empty document should produce zero Anomaly annotations");
    }

    /** German text must not throw; detection quality may vary. */
    @Test
    @DisplayName("German text does not cause an exception")
    void testGermanText() throws Exception {
        String text = "Herr Klaus Muller wohnt in der Goethestrasse 12, 60313 Frankfurt am Main.";
        createCas("de", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));

        assertDoesNotThrow(() -> composer.run(cas));
        System.out.println("German Anomaly count: " + collectAnomalies().size());
    }

    /** XMI round-trip: annotate and write to src/test/results/ for manual inspection. */
    @Test
    @DisplayName("XMI output is written to src/test/results/")
    void testXmiOutput() throws Exception {
        String text = "Maria Schmidt (m.schmidt@example.de) lives at Berliner Str. 5, 10115 Berlin.";
        createCas("en", text);

        composer.add(new DUUIRemoteDriver.Component(SERVICE_URL)
                .withParameter("mode", "placeholder"));

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

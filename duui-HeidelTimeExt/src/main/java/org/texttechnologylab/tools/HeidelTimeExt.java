package org.texttechnologylab.tools;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import de.unihd.dbs.uima.annotator.heideltime.HeidelTime;
import de.unihd.dbs.uima.types.heideltime.Timex3;
import org.apache.uima.UIMAException;
import org.apache.uima.analysis_engine.AnalysisEngine;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.cas.impl.XmiSerializationSharedData;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.factory.TypeSystemDescriptionFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.json.JSONArray;
import org.json.JSONObject;
import org.texttechnologylab.annotation.type.Time;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.StringWriter;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * DUUI wrapper for TTLab's HeidelTimeExt component.
 *
 * This wrapper uses the HeidelTimeExt repository directly via the heideltime artifact.
 * It does not depend on textimager-uima-heideltime-biofid, because that wrapper can pull
 * incompatible legacy UIMA dependencies into the runtime classpath.
 *
 * The service communicates with DUUI using XMI serialize/deserialize mode:
 * - GET  /v1/communication_layer returns communication_layer.lua
 * - GET  /v1/typesystem returns the UIMA type system
 * - GET  /v1/details/input_output returns declared input/output types
 * - POST /v1/process accepts an XMI CAS and returns the processed XMI CAS
 */
public class HeidelTimeExt {

    private static final int DEFAULT_PORT = 9714;
    private static final String COMMUNICATION_LAYER_RESOURCE = "/communication_layer.lua";
    private static final String DEFAULT_FILENAME = "duui-document";

    public static void main(String[] args) throws Exception {
        int port = getIntEnv("DUUI_PORT", DEFAULT_PORT);
        int workers = getIntEnv("DUUI_WORKERS", Runtime.getRuntime().availableProcessors());

        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/v1/communication_layer", new CommunicationLayerHandler());
        server.createContext("/v1/typesystem", new TypesystemHandler());
        server.createContext("/v1/process", new ProcessHandler());
        server.createContext("/v1/details/input_output", new IOHandler());
        server.createContext("/", new RootHandler());
        server.setExecutor(Executors.newFixedThreadPool(Math.max(1, workers)));
        server.start();

        System.out.println(HeidelTimeExt.class.getSimpleName() + " ready on port " + port + " with " + workers + " workers");
    }

    private static int getIntEnv(String key, int fallback) {
        String value = System.getenv(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static String getEnv(String key, String fallback) {
        String value = System.getenv(key);
        if (value == null || value.isBlank()) {
            return fallback;
        }
        return value.trim();
    }

    private static void writeResponse(HttpExchange exchange, int statusCode, String contentType, byte[] body) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", contentType);
        exchange.sendResponseHeaders(statusCode, body.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(body);
        }
    }

    private static void writeText(HttpExchange exchange, int statusCode, String contentType, String body) throws IOException {
        writeResponse(exchange, statusCode, contentType, body.getBytes(StandardCharsets.UTF_8));
    }

    private static void methodNotAllowed(HttpExchange exchange) throws IOException {
        writeText(exchange, 405, "text/plain; charset=utf-8", "Method not allowed");
    }

    private static class RootHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            JSONObject details = new JSONObject();
            details.put("name", "duui-heideltime-ext");
            details.put("version", System.getenv().getOrDefault("ANNOTATOR_VERSION", "1.0"));
            details.put("description", "DUUI wrapper for TTLab HeidelTimeExt");
            details.put("endpoints", new JSONArray()
                    .put("/v1/communication_layer")
                    .put("/v1/typesystem")
                    .put("/v1/details/input_output")
                    .put("/v1/process"));
            writeText(exchange, 200, "application/json; charset=utf-8", details.toString());
        }
    }

    private static class ProcessHandler implements HttpHandler {

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                methodNotAllowed(exchange);
                return;
            }

            try {
                JCas jCas = JCasFactory.createJCas();
                XmiSerializationSharedData sharedData = new XmiSerializationSharedData();

                XmiCasDeserializer.deserialize(exchange.getRequestBody(), jCas.getCas(), true, sharedData);

                ensureHeidelTimeInputAnnotations(jCas);

                AnalysisEngine analysisEngine = createPipeline(jCas);
                SimplePipeline.runPipeline(jCas, analysisEngine);
                copyTimex3ToTTLabTime(jCas);

                ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
                XmiCasSerializer.serialize(jCas.getCas(), null, outputStream, false, sharedData);

                writeResponse(exchange, 200, "application/xmi+xml", outputStream.toByteArray());
            } catch (Exception e) {
                e.printStackTrace();
                String message = "HeidelTimeExt processing failed: " + e.getMessage();
                writeText(exchange, 500, "text/plain; charset=utf-8", message);
            }
        }

        private AnalysisEngine createPipeline(JCas jCas) throws UIMAException {
            String language = getEnv("HEIDELTIME_LANGUAGE", normalizeLanguage(jCas.getDocumentLanguage()));
            String typeToProcess = getEnv("HEIDELTIME_TYPE", "narrative");
            String locale = getEnv("HEIDELTIME_LOCALE", localeForLanguage(language));
            boolean findTemponyms = Boolean.parseBoolean(getEnv("HEIDELTIME_TEMPONYMS", "true"));

            AggregateBuilder pipeline = new AggregateBuilder();
            pipeline.add(createEngineDescription(
                    HeidelTime.class,
                    "Language", language,
                    "Type", typeToProcess,
                    "locale", locale,
                    "Date", true,
                    "Time", true,
                    "Duration", true,
                    "Set", true,
                    "Temponym", findTemponyms,
                    "ConvertDurations", true,
                    "Debugging", false
            ));
            return pipeline.createAggregate();
        }

        /**
         * HeidelTime expects its own Sentence/Token types. DUUI pipelines often provide DKPro
         * Sentence/Token annotations, so we mirror them into the HeidelTime type system before
         * executing the HeidelTime annotator.
         */
        private void ensureHeidelTimeInputAnnotations(JCas jCas) {
            if (JCasUtil.exists(jCas, de.unihd.dbs.uima.types.heideltime.Sentence.class)
                    && JCasUtil.exists(jCas, de.unihd.dbs.uima.types.heideltime.Token.class)) {
                return;
            }

            List<Sentence> dkproSentences = new ArrayList<>(JCasUtil.select(jCas, Sentence.class));
            if (dkproSentences.isEmpty() && jCas.getDocumentText() != null) {
                Sentence sentence = new Sentence(jCas, 0, jCas.getDocumentText().length());
                sentence.addToIndexes();
                dkproSentences.add(sentence);
            }

            int sentenceId = 1;
            for (Sentence dkproSentence : dkproSentences) {
                de.unihd.dbs.uima.types.heideltime.Sentence heidelSentence =
                        new de.unihd.dbs.uima.types.heideltime.Sentence(
                                jCas,
                                dkproSentence.getBegin(),
                                dkproSentence.getEnd()
                        );
                heidelSentence.setFilename(DEFAULT_FILENAME);
                heidelSentence.setSentenceId(sentenceId);
                heidelSentence.addToIndexes();
                sentenceId++;
            }

            List<Token> dkproTokens = new ArrayList<>(JCasUtil.select(jCas, Token.class));
            if (dkproTokens.isEmpty()) {
                createWhitespaceTokens(jCas);
                dkproTokens = new ArrayList<>(JCasUtil.select(jCas, Token.class));
            }

            int tokenId = 1;
            for (Token dkproToken : dkproTokens) {
                de.unihd.dbs.uima.types.heideltime.Token heidelToken =
                        new de.unihd.dbs.uima.types.heideltime.Token(
                                jCas,
                                dkproToken.getBegin(),
                                dkproToken.getEnd()
                        );
                heidelToken.setFilename(DEFAULT_FILENAME);
                heidelToken.setTokenId(tokenId);
                heidelToken.setSentId(findSentenceId(dkproSentences, dkproToken));
                heidelToken.setPos("NN");
                heidelToken.addToIndexes();
                tokenId++;
            }

            ensureDct(jCas);
        }

        private void createWhitespaceTokens(JCas jCas) {
            String text = jCas.getDocumentText();
            if (text == null || text.isEmpty()) {
                return;
            }

            int tokenBegin = -1;
            for (int i = 0; i <= text.length(); i++) {
                boolean boundary = i == text.length() || Character.isWhitespace(text.charAt(i));

                if (!boundary && tokenBegin < 0) {
                    tokenBegin = i;
                }

                if (boundary && tokenBegin >= 0) {
                    Token token = new Token(jCas, tokenBegin, i);
                    token.addToIndexes();
                    tokenBegin = -1;
                }
            }
        }

        private int findSentenceId(List<Sentence> sentences, Token token) {
            for (int i = 0; i < sentences.size(); i++) {
                Sentence sentence = sentences.get(i);
                if (token.getBegin() >= sentence.getBegin() && token.getEnd() <= sentence.getEnd()) {
                    return i + 1;
                }
            }
            return 1;
        }

        private void ensureDct(JCas jCas) {
            if (JCasUtil.exists(jCas, de.unihd.dbs.uima.types.heideltime.Dct.class)) {
                return;
            }

            String today = LocalDate.now().format(DateTimeFormatter.BASIC_ISO_DATE);
            de.unihd.dbs.uima.types.heideltime.Dct dct = new de.unihd.dbs.uima.types.heideltime.Dct(jCas, 0, 0);
            dct.setFilename(DEFAULT_FILENAME);
            dct.setTimexId("dct");
            dct.setValue(today);
            dct.addToIndexes();
        }

        private String normalizeLanguage(String documentLanguage) {
            if (documentLanguage == null || documentLanguage.isBlank()) {
                return "german";
            }

            String language = documentLanguage.trim().toLowerCase();
            if (language.equals("de") || language.equals("deu") || language.equals("ger") || language.equals("german")) {
                return "german";
            }
            if (language.equals("en") || language.equals("eng") || language.equals("english")) {
                return "english";
            }
            if (language.equals("nl") || language.equals("nld") || language.equals("dut") || language.equals("dutch")) {
                return "dutch";
            }
            if (language.equals("es") || language.equals("spa") || language.equals("spanish")) {
                return "spanish";
            }
            if (language.equals("it") || language.equals("ita") || language.equals("italian")) {
                return "italian";
            }
            if (language.equals("fr") || language.equals("fra") || language.equals("fre") || language.equals("french")) {
                return "french";
            }
            if (language.equals("pt") || language.equals("por") || language.equals("portuguese")) {
                return "portuguese";
            }
            if (language.equals("ru") || language.equals("rus") || language.equals("russian")) {
                return "russian";
            }
            if (language.equals("zh") || language.equals("zho") || language.equals("chi") || language.equals("chinese")) {
                return "chinese";
            }
            if (language.equals("ar") || language.equals("ara") || language.equals("arabic")) {
                return "arabic";
            }
            if (language.equals("hr") || language.equals("hrv") || language.equals("croatian")) {
                return "croatian";
            }
            if (language.equals("et") || language.equals("est") || language.equals("estonian")) {
                return "estonian";
            }
            if (language.equals("vi") || language.equals("vie") || language.equals("vietnamese")) {
                return "vietnamese";
            }

            return language;
        }

        private String localeForLanguage(String language) {
            if ("german".equalsIgnoreCase(language)) {
                return "de_DE";
            }
            if ("english".equalsIgnoreCase(language)) {
                return "en_GB";
            }
            return "en_GB";
        }

        private void copyTimex3ToTTLabTime(JCas jCas) {
            for (Timex3 timex3 : JCasUtil.select(jCas, Timex3.class)) {
                Time time = new Time(jCas);
                time.setBegin(timex3.getBegin());
                time.setEnd(timex3.getEnd());
                time.setValue(timex3.getTimexType());
                time.setIdentifier(timex3.getTimexValue());
                time.addToIndexes();
            }
        }
    }

    private static class TypesystemHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                methodNotAllowed(exchange);
                return;
            }

            try {
                TypeSystemDescription description = TypeSystemDescriptionFactory.createTypeSystemDescription();
                StringWriter writer = new StringWriter();
                description.toXML(writer);
                writeText(exchange, 200, "application/xml; charset=utf-8", writer.toString());
            } catch (ResourceInitializationException | SAXException e) {
                e.printStackTrace();
                writeText(exchange, 500, "text/plain; charset=utf-8", "Could not create type system: " + e.getMessage());
            }
        }
    }

    private static class IOHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                methodNotAllowed(exchange);
                return;
            }

            JSONObject response = new JSONObject();
            response.put("input", new JSONArray()
                    .put(Token.class.getName())
                    .put(Sentence.class.getName()));
            response.put("output", new JSONArray()
                    .put(Timex3.class.getName())
                    .put(Time.class.getName()));

            writeText(exchange, 200, "application/json; charset=utf-8", response.toString());
        }
    }

    private static class CommunicationLayerHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                methodNotAllowed(exchange);
                return;
            }

            try (InputStream inputStream = HeidelTimeExt.class.getResourceAsStream(COMMUNICATION_LAYER_RESOURCE)) {
                if (inputStream == null) {
                    writeText(exchange, 500, "text/plain; charset=utf-8", "Missing resource: " + COMMUNICATION_LAYER_RESOURCE);
                    return;
                }
                byte[] response = inputStream.readAllBytes();
                writeResponse(exchange, 200, "text/plain; charset=utf-8", response);
            }
        }
    }
}

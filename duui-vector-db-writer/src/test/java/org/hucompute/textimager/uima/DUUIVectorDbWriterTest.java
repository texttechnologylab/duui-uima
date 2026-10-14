package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FloatArray;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.DocumentModification;
import org.texttechnologylab.annotation.MetaData;
import org.texttechnologylab.uima.type.Embedding;

import java.net.URI;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Erwartet einen laufenden Container (ohne weitere Konfiguration, alle
 * Verbindungsdaten kommen als DUUI-Parameter):
 *   docker run --rm -p 9714:9714 --add-host=host.docker.internal:host-gateway \
 *     docker.texttechnologylab.org/duui-vector-db-writer:0.0.2
 * sowie eine aus dem Container erreichbare Postgres-Instanz mit installierter
 * pgvector-Extension (CREATE EXTENSION vector;) und eine Qdrant-Instanz.
 * Wegwerf-Instanzen mit passenden Defaults: src/test/bash/test_dbs.sh start
 * Verbindungsdaten per System-Property (oder Env), Defaults siehe unten:
 *   -Dpg.connection=postgresql://host.docker.internal:5432/duui_test
 *   -Dpg.user=duui -Dpg.password=duui
 *   -Dqdrant.host=host.docker.internal -Dqdrant.port=6333
 * Prueft nur den DUUI-Roundtrip (CAS -> Service -> CAS), nicht den DB-Inhalt
 * selbst.
 */
public class DUUIVectorDbWriterTest {
    private static final String PG_CONNECTION = config("pg.connection", "PG_CONNECTION", "postgresql://host.docker.internal:5432/duui_test");
    private static final String PG_USER = config("pg.user", "PG_USER", "duui");
    private static final String PG_PASSWORD = config("pg.password", "PG_PASSWORD", "duui");
    private static final String QDRANT_HOST = config("qdrant.host", "QDRANT_HOST", "host.docker.internal");
    private static final String QDRANT_PORT = config("qdrant.port", "QDRANT_PORT", "6333");

    @Test
    public void testWriteEmbeddingsPostgres() throws Exception {
        runWriteTest("postgres");
    }

    @Test
    public void testWriteEmbeddingsQdrant() throws Exception {
        runWriteTest("qdrant");
    }

    /**
     * Alternative zum Connection-String: Host/Port/Datenbank einzeln (aus
     * derselben Testkonfiguration abgeleitet).
     */
    @Test
    public void testPostgresSeparateConnectionParameters() throws Exception {
        URI uri = URI.create(PG_CONNECTION);

        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("db_backend", "postgres");
        parameters.put("pg_host", uri.getHost());
        if (uri.getPort() != -1) {
            parameters.put("pg_port", String.valueOf(uri.getPort()));
        }
        parameters.put("pg_database", uri.getPath().replaceFirst("^/", ""));
        parameters.put("pg_user", PG_USER);
        parameters.put("pg_password", PG_PASSWORD);
        parameters.put("target_table_prefix", "test_emb");

        String comment = writeSingle(parameters, "separate-params-doc");
        assert comment.contains(", Wrote") : "Schreibvorgang mit pg_host/pg_port/pg_database fehlgeschlagen: " + comment;
    }

    /**
     * Qdrant legt das Distanzmass beim Anlegen einer Collection fest und
     * erlaubt danach keine Aenderung mehr. Erster Schreibvorgang legt die
     * Collection mit "cosine" an, zweiter fragt fuer dieselbe Collection
     * "euclid" an -- das muss der Writer ablehnen statt es stillschweigend
     * zu ignorieren (auch dann noch, wenn der Container die Collection aus
     * dem ersten Aufruf bereits im Speicher-Cache kennt).
     */
    @Test
    public void testQdrantDistanceMismatchIsRejected() throws Exception {
        String table = "test_emb_distance_mismatch";

        String firstComment = writeWithQdrantDistance(table, "cosine", "mismatch-doc-1", "false");
        assert firstComment.contains(", Wrote") : "Erster Schreibvorgang (cosine) war nicht erfolgreich: " + firstComment;

        String secondComment = writeWithQdrantDistance(table, "euclid", "mismatch-doc-2", "false");
        assert !secondComment.contains(", Wrote") : "Abweichendes Distanzmass wurde nicht abgelehnt: " + secondComment;
        assert secondComment.contains("already exists with distance")
                : "Fehlermeldung erklaert den Mismatch nicht: " + secondComment;
    }

    /**
     * Mit fail_on_error=true (Default) muss ein fehlgeschlagener
     * Schreibvorgang das Dokument in DUUI scheitern lassen.
     */
    @Test
    public void testFailOnErrorAbortsDocument() throws Exception {
        String table = "test_emb_fail_on_error";
        writeWithQdrantDistance(table, "cosine", "fail-doc-1", "false");

        assertThrows(Exception.class, () -> writeWithQdrantDistance(table, "euclid", "fail-doc-2", null));
    }

    /**
     * DocumentMetaData ohne documentId: keine Dokument-ID, also muss das
     * Dokument fehlschlagen, unabhaengig von fail_on_error.
     */
    @Test
    public void testMissingDocumentIdFails() {
        Map<String, String> parameters = connection("postgres");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("fail_on_error", "false");

        assertThrows(Exception.class, () -> writeSingle(parameters, null));
    }

    /**
     * CAS ganz ohne DocumentMetaData: DUUIComposer.run(JCas) legt dann selbst
     * eine mit documentId "UIMA-Document" an. Diese ID teilen sich alle solchen
     * Dokumente, deshalb muss der Writer sie ablehnen.
     */
    @Test
    public void testMissingDocumentMetaDataFails() {
        Map<String, String> parameters = connection("postgres");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("fail_on_error", "false");

        assertThrows(Exception.class, () -> write(parameters, jCas -> {
            MetaData model = addModel(jCas, "test-model");
            addEmbedding(jCas, model, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
        }, null, false));
    }

    /**
     * Zwei Modelle mit unterschiedlicher Dimension in einem CAS: mit
     * target_table_prefix muss jedes Modell in seine eigene Tabelle/Collection.
     */
    @Test
    public void testMultipleModelsPostgres() throws Exception {
        runMultiModelTest("postgres");
    }

    @Test
    public void testMultipleModelsQdrant() throws Exception {
        runMultiModelTest("qdrant");
    }

    /**
     * Leerer target_table_prefix: Tabelle/Collection traegt nur den
     * (backend-gerecht angepassten) Modellnamen.
     */
    @Test
    public void testEmptyPrefixPostgres() throws Exception {
        runEmptyPrefixTest("postgres", "org_model_no_prefix");
    }

    @Test
    public void testEmptyPrefixQdrant() throws Exception {
        runEmptyPrefixTest("qdrant", "org__Model-No-Prefix");
    }

    /** Ohne Praefix ergibt ein Modellname mit fuehrender Ziffer keinen gueltigen Postgres-Tabellennamen. */
    @Test
    public void testEmptyPrefixWithDigitModelPostgres() throws Exception {
        Map<String, String> parameters = connection("postgres");
        parameters.put("target_table_prefix", "");
        parameters.put("fail_on_error", "false");

        String comment = write(parameters, jCas -> {
            MetaData model = addModel(jCas, "123-model");
            addEmbedding(jCas, model, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
        }, "digit-model-doc");
        assert comment.contains("not a valid Postgres identifier") : "Unerwartete Antwort: " + comment;
    }

    private void runEmptyPrefixTest(String dbBackend, String expectedTarget) throws Exception {
        Map<String, String> parameters = connection(dbBackend);
        parameters.put("target_table_prefix", "");

        String comment = write(parameters, jCas -> {
            MetaData model = addModel(jCas, "org/Model-No-Prefix");
            addEmbedding(jCas, model, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
        }, "empty-prefix-doc-" + dbBackend);
        assert comment.contains(", Wrote") && comment.endsWith(": " + expectedTarget)
                : "Unerwarteter Tabellen-/Collection-Name: " + comment;
    }

    /**
     * Der Container hat keine eigene Konfiguration: ohne Verbindungsparameter
     * muss der Writer einen verstaendlichen Fehler liefern.
     */
    @Test
    public void testMissingConnectionParametersPostgres() throws Exception {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("db_backend", "postgres");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("fail_on_error", "false");

        String comment = writeSingle(parameters, "missing-params-postgres");
        assert comment.contains("Missing Postgres connection parameter") : "Unerwartete Antwort: " + comment;
    }

    @Test
    public void testMissingConnectionParametersQdrant() throws Exception {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("db_backend", "qdrant");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("fail_on_error", "false");

        String comment = writeSingle(parameters, "missing-params-qdrant");
        assert comment.contains("Missing Qdrant connection parameter") : "Unerwartete Antwort: " + comment;
    }

    /**
     * Die uebergebenen Verbindungsparameter muessen tatsaechlich verwendet
     * werden: mit absichtlich falschen Werten muss der Schreibvorgang scheitern.
     */
    @Test
    public void testWrongPasswordPostgres() throws Exception {
        Map<String, String> parameters = connection("postgres");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("pg_password", "definitely-wrong-password");
        parameters.put("fail_on_error", "false");

        String comment = writeSingle(parameters, "wrong-password-postgres");
        assert comment.contains(", Error:") : "Falsches pg_password wurde nicht verwendet: " + comment;
    }

    @Test
    public void testWrongPortQdrant() throws Exception {
        Map<String, String> parameters = connection("qdrant");
        parameters.put("target_table_prefix", "test_emb");
        parameters.put("qdrant_port", "1");
        parameters.put("fail_on_error", "false");

        String comment = writeSingle(parameters, "wrong-port-qdrant");
        assert comment.contains(", Error:") : "Falscher qdrant_port wurde nicht verwendet: " + comment;
    }

    private String writeWithQdrantDistance(String targetTable, String qdrantDistance, String docId, String failOnError) throws Exception {
        Map<String, String> parameters = connection("qdrant");
        parameters.put("target_table", targetTable);
        parameters.put("qdrant_distance", qdrantDistance);
        if (failOnError != null) {
            parameters.put("fail_on_error", failOnError);
        }

        String comment = writeSingle(parameters, docId);
        System.out.println("Writer-Antwort (qdrant, distance=" + qdrantDistance + "): " + comment);
        return comment;
    }

    private void runWriteTest(String dbBackend) throws Exception {
        Map<String, String> parameters = connection(dbBackend);
        parameters.put("target_table_prefix", "test_emb");

        String comment = write(parameters, jCas -> {
            MetaData model = addModel(jCas, "test-model");
            addEmbedding(jCas, model, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
            addEmbedding(jCas, model, 19, 41, new float[]{0.4f, 0.5f, 0.6f});
        }, "test-doc-" + dbBackend);
        System.out.println("Writer-Antwort (" + dbBackend + "): " + comment);
        assert comment.contains(", Wrote 5 ") : "Schreibvorgang war nicht erfolgreich: " + comment;
    }

    private void runMultiModelTest(String dbBackend) throws Exception {
        Map<String, String> parameters = connection(dbBackend);
        parameters.put("target_table_prefix", "test_emb_multi");

        String comment = write(parameters, jCas -> {
            MetaData modelA = addModel(jCas, "org/Model-A");
            addEmbedding(jCas, modelA, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
            addEmbedding(jCas, modelA, 19, 41, new float[]{0.4f, 0.5f, 0.6f});

            MetaData modelB = addModel(jCas, "org/Model-B");
            addEmbedding(jCas, modelB, 0, 18, new float[]{0.1f, 0.2f, 0.3f, 0.4f});
        }, "multi-doc-" + dbBackend);
        System.out.println("Writer-Antwort (" + dbBackend + ", multi-model): " + comment);

        // model-a: 2 Saetze + 3 Aggregate, model-b: 1 Satz + 3 Aggregate
        assert comment.contains(", Wrote 9 ") : "Unerwartete Anzahl geschriebener Eintraege: " + comment;

        // Postgres: SQL-Identifier, kleingeschrieben. Qdrant: Modellname
        // unveraendert, nur das verbotene "/" wird zu "__".
        String expectedA = dbBackend.equals("postgres") ? "test_emb_multi_org_model_a" : "test_emb_multi_org__Model-A";
        String expectedB = dbBackend.equals("postgres") ? "test_emb_multi_org_model_b" : "test_emb_multi_org__Model-B";
        assert comment.contains(expectedA) && comment.contains(expectedB)
                : "Modelle wurden nicht in die erwarteten Tabellen/Collections geschrieben: " + comment;
    }

    /** Backend und dessen Verbindungsparameter aus der Testkonfiguration. */
    private static Map<String, String> connection(String dbBackend) {
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("db_backend", dbBackend);
        if (dbBackend.equals("postgres")) {
            parameters.put("pg_connection_string", PG_CONNECTION);
            parameters.put("pg_user", PG_USER);
            parameters.put("pg_password", PG_PASSWORD);
        } else {
            parameters.put("qdrant_host", QDRANT_HOST);
            parameters.put("qdrant_port", QDRANT_PORT);
        }
        return parameters;
    }

    private String writeSingle(Map<String, String> parameters, String docId) throws Exception {
        return write(parameters, jCas -> {
            MetaData model = addModel(jCas, "test-model");
            addEmbedding(jCas, model, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
        }, docId);
    }

    /** docId == null: DocumentMetaData ohne documentId. */
    private String write(Map<String, String> parameters, Consumer<JCas> fill, String docId) throws Exception {
        return write(parameters, fill, docId, true);
    }

    /** withDocumentMetaData == false: CAS ganz ohne DocumentMetaData. */
    private String write(Map<String, String> parameters, Consumer<JCas> fill, String docId,
                         boolean withDocumentMetaData) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component("http://localhost:9714");
        for (Map.Entry<String, String> parameter : parameters.entrySet()) {
            component.withParameter(parameter.getKey(), parameter.getValue());
        }
        composer.add(component.build().withTimeout(30000L));

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText("Das ist ein Test. Das ist noch ein Test.");
        jCas.setDocumentLanguage("de");

        if (withDocumentMetaData) {
            DocumentMetaData meta = DocumentMetaData.create(jCas);
            if (docId != null) {
                meta.setDocumentId(docId);
            }
            meta.addToIndexes();
        }

        fill.accept(jCas);

        try {
            composer.run(jCas);
        } finally {
            composer.shutdown();
        }

        List<DocumentModification> modifications =
                new java.util.ArrayList<>(JCasUtil.select(jCas, DocumentModification.class));
        assert !modifications.isEmpty() : "Kein DocumentModification-Eintrag vom Writer erhalten";
        return modifications.get(0).getComment();
    }

    private static String config(String property, String env, String defaultValue) {
        String value = System.getProperty(property);
        if (value == null) {
            value = System.getenv(env);
        }
        return value != null ? value : defaultValue;
    }

    private static MetaData addModel(JCas jCas, String modelName) {
        MetaData modelMeta = new MetaData(jCas);
        modelMeta.setSource(modelName);
        modelMeta.addToIndexes();
        return modelMeta;
    }

    private static void addEmbedding(JCas jCas, MetaData modelMeta, int begin, int end, float[] vector) {
        Embedding embedding = new Embedding(jCas, begin, end);
        embedding.setModelReference(modelMeta);
        embedding.setEmbedding(new FloatArray(jCas, vector.length));
        for (int i = 0; i < vector.length; i++) {
            embedding.getEmbedding().set(i, vector[i]);
        }
        embedding.addToIndexes();
    }
}

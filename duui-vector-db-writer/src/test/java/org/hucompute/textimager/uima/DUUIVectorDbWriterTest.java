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

import java.util.List;

/**
 * Erwartet einen laufenden Container:
 *   docker run --rm -p 9714:9714 \
 *     -e DUUI_VECTOR_DB_WRITER_DB_HOST=host.docker.internal \
 *     -e DUUI_VECTOR_DB_WRITER_DB_NAME=duui_test \
 *     -e DUUI_VECTOR_DB_WRITER_DB_USER=duui \
 *     -e DUUI_VECTOR_DB_WRITER_DB_PASSWORD=duui \
 *     -e DUUI_VECTOR_DB_WRITER_QDRANT_HOST=host.docker.internal \
 *     -e DUUI_VECTOR_DB_WRITER_QDRANT_PORT=6333 \
 *     docker.texttechnologylab.org/duui-vector-db-writer:0.0.1
 * sowie eine lokal erreichbare Postgres-Instanz mit installierter
 * pgvector-Extension (CREATE EXTENSION vector;) und eine lokal erreichbare
 * Qdrant-Instanz. Prueft nur den DUUI-Roundtrip (CAS -> Service -> CAS),
 * nicht den DB-Inhalt selbst.
 */
public class DUUIVectorDbWriterTest {
    @Test
    public void testWriteEmbeddingsPostgres() throws Exception {
        runWriteTest("postgres");
    }

    @Test
    public void testWriteEmbeddingsQdrant() throws Exception {
        runWriteTest("qdrant");
    }

    private void runWriteTest(String dbBackend) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withParameter("db_backend", dbBackend)
                        .withParameter("target_table_prefix", "test_emb")
                        .build()
                        .withTimeout(30000L)
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText("Das ist ein Test. Das ist noch ein Test.");
        jCas.setDocumentLanguage("de");

        DocumentMetaData meta = DocumentMetaData.create(jCas);
        meta.setDocumentId("test-doc-" + dbBackend);
        meta.addToIndexes();

        MetaData modelMeta = new MetaData(jCas);
        modelMeta.setSource("test-model");
        modelMeta.addToIndexes();

        addEmbedding(jCas, modelMeta, 0, 18, new float[]{0.1f, 0.2f, 0.3f});
        addEmbedding(jCas, modelMeta, 19, 41, new float[]{0.4f, 0.5f, 0.6f});

        composer.run(jCas);
        composer.shutdown();

        List<DocumentModification> modifications =
                new java.util.ArrayList<>(JCasUtil.select(jCas, DocumentModification.class));
        assert !modifications.isEmpty() : "Kein DocumentModification-Eintrag vom Writer erhalten";

        String comment = modifications.get(0).getComment();
        System.out.println("Writer-Antwort (" + dbBackend + "): " + comment);
        assert comment.startsWith("Wrote") : "Schreibvorgang war nicht erfolgreich: " + comment;
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

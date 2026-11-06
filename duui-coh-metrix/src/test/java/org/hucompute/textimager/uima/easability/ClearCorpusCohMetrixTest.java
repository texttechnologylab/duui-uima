package org.hucompute.textimager.uima.easability;

import org.dkpro.core.io.xmi.XmiWriter;
import org.hucompute.textimager.uima.cohmetrix.CohMetrixUdPipelineTest;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUICollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * CLEAR corpus -> the default pipeline of Coh-Metrix -> one gzipped XMI per excerpt, named by the corpus' excerpt ID:
 * the reference corpus of the CLEAR Text Easability models (README, "Text Easability models"). English from the
 * original corpus ({@link DUUIClearCorpusReader}), German from its translation ({@link DUUIClearTranslationReader}).
 *
 * <p>The paragraphs come from the readers (CLEAR's own: one per line; the translation reader skips excerpts whose
 * translation has another number of paragraphs). After them the default pipeline of
 * {@link CohMetrixUdPipelineTest#addAfterParagraphs} (sentences, tokens, Stanza, spaCy vectors, removal of the plain
 * layers, Coh-Metrix), all as remote services: the system properties of {@link CohMetrixUdPipelineTest}
 * ({@code wtpsplit.url}, {@code somajo.url}, {@code stanza.url}, {@code spacy.url}, {@code remove.url},
 * {@code cohmetrix.url}). The Coh-Metrix service must run with GermaNet and the resources supplied by the user mounted
 * (README, "Resources"): they are inputs of the models; without them the inputs are undefined.
 *
 * <p>Output: {@code <CLEAR_DIR>/xmi_en_<version>} and {@code <CLEAR_DIR>/xmi_de_<translation>_<version>}, the version
 * of the Coh-Metrix service (checked against {@link #COH_METRIX_VERSION}); the XMIs of earlier versions stay. Test
 * runs: {@code -Dclear.limit=N} (only the first N excerpts) and {@code -Dclear.target=DIR} (output directory instead).
 */
public class ClearCorpusCohMetrixTest {

    /** The Coh-Metrix version the reference corpus is made with; the service must report it. */
    static final String COH_METRIX_VERSION = "0.6.0";

    /** Corpus, translations and processed corpora (not in the repository; download_resources.py --clear). */
    static final Path CLEAR_DIR = Paths.get("/storage/projects/CORE/projects2/coh_metrix/clear_corpus");
    private static final Path CLEAR_CSV = CLEAR_DIR.resolve("CLEAR_corpus_final.csv");

    /** Translation directory (one JSON per excerpt), default kolibri-1-q4km. */
    private static final String TRANSLATION_PROPERTY = "clear.translation";
    private static final String DEFAULT_TRANSLATION = CLEAR_DIR.resolve("translation_kolibri-1-q4km").toString();

    // test runs: the first N excerpts only, and another output directory
    private static final int LIMIT = Integer.getInteger("clear.limit", 0);
    private static final String TARGET = System.getProperty("clear.target");

    // Without the PC indices: the reference corpus is the input of the PCA, so it must not depend on an earlier
    // model (easability/pca.py)
    private static final String EXCLUDE_INDICES = "Text Easability Principal Component Scores";

    @Test
    public void runPipeline() throws Exception {
        run(new DUUIClearCorpusReader(CLEAR_CSV, "en", LIMIT), target(CLEAR_DIR.resolve("xmi_en_" + COH_METRIX_VERSION)));
    }

    /** Output: {@code <CLEAR_DIR>/xmi_de_<translation directory without "translation_">_<version>}. */
    @Test
    public void runPipelineGermanTranslation() throws Exception {
        Path translation = translationDirectory();
        run(new DUUIClearTranslationReader(translation, CLEAR_CSV, "de", LIMIT), target(germanXmiLocation(translation)));
    }

    /** {@value #TRANSLATION_PROPERTY}, default {@value #DEFAULT_TRANSLATION}. */
    static Path translationDirectory() {
        return Paths.get(System.getProperty(TRANSLATION_PROPERTY, DEFAULT_TRANSLATION));
    }

    /** {@code <CLEAR_DIR>/xmi_de_<translation directory without "translation_">_<version>}. */
    static Path germanXmiLocation(Path translation) {
        return CLEAR_DIR.resolve("xmi_de_" + translation.getFileName().toString().replaceFirst("^translation_", "")
                + "_" + COH_METRIX_VERSION);
    }

    private static Path target(Path location) {
        return TARGET != null ? Paths.get(TARGET) : location;
    }

    /** Fails unless the Coh-Metrix service reports {@link #COH_METRIX_VERSION} (it names the output directory). */
    private static void checkCohMetrixVersion() throws Exception {
        String url = System.getProperty("cohmetrix.url", "http://localhost:9802");
        try (HttpClient client = HttpClient.newHttpClient()) {
            String body = client.send(HttpRequest.newBuilder(URI.create(url + "/v1/documentation")).GET().build(),
                    HttpResponse.BodyHandlers.ofString()).body();
            String version = new JSONObject(body).getString("version");
            if (!COH_METRIX_VERSION.equals(version)) {
                throw new IllegalStateException(url + " runs Coh-Metrix " + version + ", expected " + COH_METRIX_VERSION);
            }
        }
    }

    private static void run(DUUICollectionReader reader, Path target) throws Exception {
        checkCohMetrixVersion();
        DUUIAsynchronousProcessor processor = new DUUIAsynchronousProcessor(reader);

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIUIMADriver(), new DUUIRemoteDriver());

        DUUIRemoteDriver.Component cohMetrix = CohMetrixUdPipelineTest.cohMetrix()
                .withParameter("exclude_indices", EXCLUDE_INDICES);
        CohMetrixUdPipelineTest.addAfterParagraphs(composer, cohMetrix);

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, target.toString(),
                XmiWriter.PARAM_USE_DOCUMENT_ID, true,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, false,
                XmiWriter.PARAM_VERSION, "1.1",
                XmiWriter.PARAM_COMPRESSION, "GZIP"
        )).build());

        composer.run(processor, "clear-corpus-cohmetrix");
        composer.shutdown();
    }
}

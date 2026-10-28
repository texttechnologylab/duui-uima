package org.hucompute.textimager.uima.easability;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.uima.cas.CAS;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPipelineComponent;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUICollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.DockerUnifiedUIMAInterface.segmentation.DUUISegmentationStrategyByAnnotation;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * CLEAR corpus -> spaCy, run on each paragraph separately -> Coh-Metrix -> one gzipped XMI per excerpt, named by the
 * corpus' excerpt ID. Basis for the Text Easability reference corpora: English from the original corpus
 * ({@link DUUIClearCorpusReader}), German from its translation ({@link DUUIClearTranslationReader}).
 *
 * <p>The paragraphs come from the readers, so no Syntok step. spaCy is segmented by paragraph so that its sentences
 * cannot cross a paragraph boundary: Coh-Metrix assigns sentences to paragraphs with {@code selectCovered} and drops
 * sentences that are not completely inside a paragraph.
 *
 * <p>German needs GermaNet ({@link GermanetCohMetrixContainer}): without it the polysemy and verb overlap inputs are
 * undefined for every excerpt.
 */
public class ClearCorpusCohMetrixTest {

    // TODO replace with the Docker image before committing/releasing
    private static final String SPACY_URL = "http://geltlin.hucompute.org:9727";
    // two-layer labels (docs/label-scheme.md); built by src/main/bash/docker_build.sh
    private static final String COHMETRIX_IMAGE = "docker.texttechnologylab.org/duui-coh-metrix:0.2.0";

    /** Corpus, translations and processed corpora (not in the repository, see README). */
    static final Path CLEAR_DIR = Paths.get("/storage/projects/CORE/projects2/coh_metrix/clear_corpus");
    private static final Path CLEAR_CSV = CLEAR_DIR.resolve("CLEAR_corpus_final.csv");

    /** Translation directory (translation_eval/translate_clear.py output), default kolibri-1-q4km. */
    private static final String TRANSLATION_PROPERTY = "clear.translation";
    private static final String DEFAULT_TRANSLATION = CLEAR_DIR.resolve("translation_kolibri-1-q4km").toString();

    // Without the PC indices: the reference corpus is the input of the PCA, so it must not depend on an earlier
    // model (easability/pca.py)
    private static final String EXCLUDE_INDICES = "Text Easability Principal Component Scores";

    @Test
    public void runPipeline() throws Exception {
        DUUIPipelineComponent cohMetrix = new DUUIDockerDriver.Component(COHMETRIX_IMAGE)
                .withParameter("exclude_indices", EXCLUDE_INDICES)
                .withScale(1)
                .build()
                .withTimeout(500L);
        run(new DUUIClearCorpusReader(CLEAR_CSV, "en"), cohMetrix, CLEAR_DIR.resolve("xmi_en"));
    }

    /** Output: {@code <CLEAR_DIR>/xmi_de_<translation directory without "translation_">}. */
    @Test
    public void runPipelineGermanTranslation() throws Exception {
        Path translation = translationDirectory();
        try (GermanetCohMetrixContainer container = new GermanetCohMetrixContainer(COHMETRIX_IMAGE)) {
            DUUIPipelineComponent cohMetrix = new DUUIRemoteDriver.Component(container.url())
                    .withParameter("exclude_indices", EXCLUDE_INDICES)
                    .withScale(1)
                    .build()
                    .withTimeout(500L);
            run(new DUUIClearTranslationReader(translation, CLEAR_CSV, "de"), cohMetrix,
                    germanXmiLocation(translation));
        }
    }

    /** {@value #TRANSLATION_PROPERTY}, default {@value #DEFAULT_TRANSLATION}. */
    static Path translationDirectory() {
        return Paths.get(System.getProperty(TRANSLATION_PROPERTY, DEFAULT_TRANSLATION));
    }

    /** {@code <CLEAR_DIR>/xmi_de_<translation directory without "translation_">}. */
    static Path germanXmiLocation(Path translation) {
        return CLEAR_DIR.resolve("xmi_de_"
                + translation.getFileName().toString().replaceFirst("^translation_", ""));
    }

    private static void run(DUUICollectionReader reader, DUUIPipelineComponent cohMetrix, Path target)
            throws Exception {
        DUUIAsynchronousProcessor processor = new DUUIAsynchronousProcessor(reader);

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        composer.addDriver(new DUUIUIMADriver(), new DUUIDockerDriver(), new DUUIRemoteDriver());

        // One paragraph per segment (see class comment). No parameters: the service chooses its default model for
        // the document language (en -> en_core_web_sm).
        DUUISegmentationStrategyByAnnotation paragraphs = new DUUISegmentationStrategyByAnnotation()
                .withSegmentationClass(Paragraph.class)
                .withView(CAS.NAME_DEFAULT_SOFA)
                .withStrictSegmentationClass(true)
                .withMaxCharsPerSegment(1000000000)
                .withMaxAnnotationsPerSegment(1)
                .withPrintStatistics(true);
        composer.add(new DUUIRemoteDriver.Component(SPACY_URL)
                .withScale(1)
                .withSegmentationStrategy(paragraphs)
                .build()
                .withTimeout(1000L));

        composer.add(cohMetrix);

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

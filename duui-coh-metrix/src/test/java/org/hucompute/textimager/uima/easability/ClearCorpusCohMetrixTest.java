package org.hucompute.textimager.uima.easability;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.uima.cas.CAS;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.DockerUnifiedUIMAInterface.segmentation.DUUISegmentationStrategyByAnnotation;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * English CLEAR corpus (see {@link DUUIClearCorpusReader}) -> spaCy, run on each paragraph separately ->
 * Coh-Metrix -> one gzipped XMI per excerpt, named by the corpus' excerpt ID. Basis for the Text Easability
 * reference corpus.
 *
 * <p>The paragraphs come from the reader (one per excerpt line), so no Syntok step. spaCy is segmented by paragraph
 * so that its sentences cannot cross a paragraph boundary: Coh-Metrix assigns sentences to paragraphs with
 * {@code selectCovered} and drops sentences that are not completely inside a paragraph.
 */
public class ClearCorpusCohMetrixTest {

    // TODO replace with the Docker image before committing/releasing
    private static final String SPACY_URL = "http://geltlin.hucompute.org:9727";
    // TODO replace with the released image; local build of the working tree (labels of a8dd58b/94c461c)
    private static final String COHMETRIX_IMAGE = "duui-coh-metrix:easability-dev";

    @Test
    public void runPipeline() throws Exception {
        Path sourceLocation = Paths.get("clear_corpus/CLEAR_corpus_final.csv");
        Path targetLocation = Paths.get("clear_corpus/xmi_en");

        DUUIAsynchronousProcessor processor = new DUUIAsynchronousProcessor(
                new DUUIClearCorpusReader(sourceLocation, "en")
        );

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

        // Without the PC indices: the reference corpus is the input of the PCA, so it must not depend on an earlier
        // model (easability/pca.py)
        composer.add(new DUUIDockerDriver.Component(COHMETRIX_IMAGE)
                .withParameter("exclude_indices", "Text Easability Principal Component Scores")
                .withScale(1)
                .build()
                .withTimeout(500L));

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, targetLocation.toString(),
                XmiWriter.PARAM_USE_DOCUMENT_ID, true,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, false,
                XmiWriter.PARAM_VERSION, "1.1",
                XmiWriter.PARAM_COMPRESSION, "GZIP"
        )).build());

        composer.run(processor, "clear-corpus-cohmetrix");
    }
}

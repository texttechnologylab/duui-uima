package org.hucompute.textimager.uima.easability;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.Feature;
import org.apache.uima.cas.FeatureStructure;
import org.apache.uima.cas.Type;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.metadata.TypeSystemDescription;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPipelineComponent;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.io.InputStream;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks the Text Easability components (PC*) of the Coh-Metrix component against the reference scores of the PCA
 * ({@code easability/pca.py}): every 100th CLEAR excerpt of {@link ClearCorpusCohMetrixTest}'s output is processed
 * again (previous Coh-Metrix output removed), and its z scores must equal the PCA's component scores. A second run
 * with only {@link #SELECTED} in {@code include_indices} must output only these, with the same values.
 *
 * <p>German: the translated corpus ({@link ClearCorpusCohMetrixTest#translationDirectory()}), with GermaNet
 * ({@link GermanetCohMetrixContainer}) as for the reference corpus.
 */
public class ClearCorpusTextEasabilityTest {

    // two-layer labels (docs/label-scheme.md); built by src/main/bash/docker_build.sh
    private static final String COHMETRIX_IMAGE = "docker.texttechnologylab.org/duui-coh-metrix:0.2.0";

    private static final int SAMPLE_STEP = 100;
    private static final double TOLERANCE = 1e-6;

    private static final String INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index";
    private static final String META_TYPE = "org.texttechnologylab.annotation.AnnotatorMetaData";
    private static final String TYPE_NAME = "Text Easability Principal Component Scores";
    private static final List<String> SELECTED = List.of("PCNARz", "PCSYNp");

    @Test
    public void componentScoresMatchPca() throws Exception {
        check(ClearCorpusCohMetrixTest.CLEAR_DIR.resolve("xmi_en"), Paths.get("easability/results/clear_en_pca/scores.csv"),
                "_REFIT_CLEAR", COHMETRIX_IMAGE, false);
    }

    @Test
    public void germanComponentScoresMatchPca() throws Exception {
        Path translation = ClearCorpusCohMetrixTest.translationDirectory();
        String name = translation.getFileName().toString().replaceFirst("^translation_", "");
        try (GermanetCohMetrixContainer container = new GermanetCohMetrixContainer(COHMETRIX_IMAGE)) {
            check(ClearCorpusCohMetrixTest.germanXmiLocation(translation),
                    Paths.get("easability/results/clear_de_" + name + "_pca/scores.csv"),
                    "_REFIT_CLEARtr", container.url(), true);
        }
    }

    /** component: a Docker image, or the URL of a running component if remote. */
    private static void check(Path xmiLocation, Path scoresFile, String internalSuffix, String component,
                              boolean remote) throws Exception {
        List<CSVRecord> scores;
        try (Reader in = Files.newBufferedReader(scoresFile, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(in)) {
            scores = parser.getRecords();
        }

        DUUIComposer composer = composer(component, remote, null);
        // only some PC indices selected: their inputs are computed, but not output
        DUUIComposer selectedComposer = composer(component, remote, String.join(",", SELECTED));

        try {
            // the CAS needs the type system of the components (Coh-Metrix index types)
            TypeSystemDescription typeSystem = composer.instantiate_pipeline();
            selectedComposer.instantiate_pipeline();

            for (int i = 0; i < scores.size(); i += SAMPLE_STEP) {
                CSVRecord expected = scores.get(i);
                String id = expected.get("document_id");

                JCas jCas = load(typeSystem, xmiLocation, id);
                composer.run(jCas);

                Map<String, FeatureStructure> components = components(jCas.getCas());
                assertEquals(16, components.size(), id);

                JCas selectedJCas = load(typeSystem, xmiLocation, id);
                selectedComposer.run(selectedJCas);
                assertEquals(SELECTED.size(), indexed(selectedJCas.getCas(),
                        selectedJCas.getTypeSystem().getType(INDEX_TYPE)).size(), id + " output indices");
                Map<String, FeatureStructure> selected = components(selectedJCas.getCas());
                for (String label : SELECTED) {
                    assertNull(string(selected.get(label), "error"), id + " " + label);
                    assertEquals(value(components.get(label)), value(selected.get(label)), 0, id + " " + label);
                }
                for (String header : expected.getParser().getHeaderNames()) {
                    if (!header.startsWith("PC")) {
                        continue;
                    }
                    String percentileLabel = header.substring(0, header.length() - 1) + "p";
                    FeatureStructure z = components.get(header);
                    FeatureStructure p = components.get(percentileLabel);
                    assertNull(string(z, "error"), id + " " + header);
                    assertEquals(header, string(z, "labelV3"), id);
                    assertEquals(header + internalSuffix, string(z, "labelTTLab"), id);
                    assertEquals(Double.parseDouble(expected.get(header)), value(z), TOLERANCE, id + " " + header);
                    assertTrue(value(p) >= 0 && value(p) <= 100, id + " " + percentileLabel);
                }
                System.out.println(id + ": " + components.size() + " Text Easability indices match, "
                        + SELECTED + " alone identical");
            }
        } finally {
            composer.shutdown();
            selectedComposer.shutdown();
        }
    }

    private static DUUIComposer composer(String component, boolean remote, String includeIndices) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        DUUIPipelineComponent pipelineComponent;
        if (remote) {
            composer.addDriver(new DUUIRemoteDriver());
            DUUIRemoteDriver.Component remoteComponent = new DUUIRemoteDriver.Component(component).withScale(1);
            if (includeIndices != null) {
                remoteComponent.withParameter("include_indices", includeIndices);
            }
            pipelineComponent = remoteComponent.build();
        } else {
            composer.addDriver(new DUUIDockerDriver());
            DUUIDockerDriver.Component dockerComponent = new DUUIDockerDriver.Component(component).withScale(1);
            if (includeIndices != null) {
                dockerComponent.withParameter("include_indices", includeIndices);
            }
            pipelineComponent = dockerComponent.build();
        }
        composer.add(pipelineComponent.withTimeout(500L));
        return composer;
    }

    /** The processed excerpt without its previous Coh-Metrix output. */
    private static JCas load(TypeSystemDescription typeSystem, Path xmiLocation, String id) throws Exception {
        JCas jCas = JCasFactory.createJCas(typeSystem);
        try (InputStream in = new GZIPInputStream(Files.newInputStream(xmiLocation.resolve(id + ".xmi.gz")))) {
            XmiCasDeserializer.deserialize(in, jCas.getCas(), true);
        }
        removePreviousCohMetrixOutput(jCas.getCas());
        return jCas;
    }

    /** Text Easability indices by labelV3. */
    private static Map<String, FeatureStructure> components(CAS cas) {
        Type indexType = cas.getTypeSystem().getType(INDEX_TYPE);
        Map<String, FeatureStructure> components = new LinkedHashMap<>();
        for (FeatureStructure index : indexed(cas, indexType)) {
            if (TYPE_NAME.equals(string(index, "typeName"))) {
                components.put(string(index, "labelV3"), index);
            }
        }
        return components;
    }

    private static void removePreviousCohMetrixOutput(CAS cas) {
        Type indexType = cas.getTypeSystem().getType(INDEX_TYPE);
        List<FeatureStructure> oldIndices = indexed(cas, indexType);
        Set<FeatureStructure> oldIndexSet = new HashSet<>(oldIndices);

        Type metaType = cas.getTypeSystem().getType(META_TYPE);
        Feature reference = metaType.getFeatureByBaseName("reference");
        for (FeatureStructure meta : indexed(cas, metaType)) {
            if (oldIndexSet.contains(meta.getFeatureValue(reference))) {
                cas.removeFsFromIndexes(meta);
            }
        }
        oldIndices.forEach(cas::removeFsFromIndexes);
    }

    private static List<FeatureStructure> indexed(CAS cas, Type type) {
        List<FeatureStructure> result = new ArrayList<>();
        cas.getIndexRepository().getAllIndexedFS(type).forEachRemaining(result::add);
        return result;
    }

    private static String string(FeatureStructure fs, String feature) {
        return fs.getStringValue(fs.getType().getFeatureByBaseName(feature));
    }

    private static double value(FeatureStructure fs) {
        return fs.getDoubleValue(fs.getType().getFeatureByBaseName("value"));
    }
}

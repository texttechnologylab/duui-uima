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
 * Checks the Text Easability components of the models fitted on CLEAR (records {@code PC*_REFIT_CLEAR}, German
 * {@code PC*_REFIT_CLEARtr}) against the reference scores of the PCA ({@code easability/pca.py}): every 100th excerpt of
 * {@link ClearCorpusCohMetrixTest}'s output (the reference corpus, {@code xmi_en_<version>} / {@code xmi_de_..._<version>})
 * is processed again by the running Coh-Metrix service (its earlier output removed), and the z scores of its CLEAR
 * records must equal the PCA's component scores of that excerpt. A second run with only {@link #SELECTED} in
 * {@code include_indices} must output only these records, with the same values.
 *
 * <p>The service ({@code -Dcohmetrix.url}, as for {@link ClearCorpusCohMetrixTest}) must run with GermaNet and the
 * resources supplied by the user mounted, as for the reference corpus; otherwise the inputs differ and the records are
 * None with an error. Until 2026-11-05 this test ran the Docker image of 0.2.0 on the 0.2.0 corpus.
 */
public class ClearCorpusTextEasabilityTest {

    private static final String COH_METRIX_URL = System.getProperty("cohmetrix.url", "http://localhost:9802");

    private static final int SAMPLE_STEP = 100;
    private static final double TOLERANCE = 1e-6;

    private static final String INDEX_TYPE = "org.texttechnologylab.uima.type.cohmetrix.Index";
    private static final String META_TYPE = "org.texttechnologylab.annotation.AnnotatorMetaData";
    /** Public labels of the records selected alone (with the language's internal suffix). */
    private static final List<String> SELECTED = List.of("PCNARz", "PCSYNp");

    @Test
    public void componentScoresMatchPca() throws Exception {
        check(ClearCorpusCohMetrixTest.CLEAR_DIR.resolve("xmi_en_" + ClearCorpusCohMetrixTest.COH_METRIX_VERSION),
                Paths.get("easability/results/clear_en_pca/scores.csv"), "_REFIT_CLEAR");
    }

    @Test
    public void germanComponentScoresMatchPca() throws Exception {
        Path translation = ClearCorpusCohMetrixTest.translationDirectory();
        String name = translation.getFileName().toString().replaceFirst("^translation_", "");
        check(ClearCorpusCohMetrixTest.germanXmiLocation(translation),
                Paths.get("easability/results/clear_de_" + name + "_pca/scores.csv"), "_REFIT_CLEARtr");
    }

    private static void check(Path xmiLocation, Path scoresFile, String internalSuffix) throws Exception {
        List<CSVRecord> scores;
        try (Reader in = Files.newBufferedReader(scoresFile, StandardCharsets.UTF_8);
             CSVParser parser = CSVFormat.DEFAULT.builder().setHeader().setSkipHeaderRecord(true).build().parse(in)) {
            scores = parser.getRecords();
        }

        List<String> selectedLabels = SELECTED.stream().map(label -> label + internalSuffix).toList();
        DUUIComposer composer = composer(null);
        // only some CLEAR records selected (by internal label): their inputs are computed, but not output
        DUUIComposer selectedComposer = composer(String.join(",", selectedLabels));

        try {
            // the CAS needs the type system of the components (Coh-Metrix index types)
            TypeSystemDescription typeSystem = composer.instantiate_pipeline();
            selectedComposer.instantiate_pipeline();

            for (int i = 0; i < scores.size(); i += SAMPLE_STEP) {
                CSVRecord expected = scores.get(i);
                String id = expected.get("document_id");

                JCas jCas = load(typeSystem, xmiLocation, id);
                composer.run(jCas);
                Map<String, FeatureStructure> records = byInternalLabel(jCas.getCas());

                JCas selectedJCas = load(typeSystem, xmiLocation, id);
                selectedComposer.run(selectedJCas);
                Map<String, FeatureStructure> selected = byInternalLabel(selectedJCas.getCas());
                assertEquals(new HashSet<>(selectedLabels), selected.keySet(), id + " output records");
                for (String label : selectedLabels) {
                    assertNull(string(selected.get(label), "error"), id + " " + label);
                    assertEquals(value(records.get(label)), value(selected.get(label)), 0, id + " " + label);
                }

                int checked = 0;
                for (String header : expected.getParser().getHeaderNames()) {
                    if (!header.startsWith("PC")) {
                        continue;
                    }
                    String percentile = header.substring(0, header.length() - 1) + "p";
                    FeatureStructure z = records.get(header + internalSuffix);
                    FeatureStructure p = records.get(percentile + internalSuffix);
                    assertTrue(z != null && p != null, id + " " + header + internalSuffix + " output");
                    assertNull(string(z, "error"), id + " " + header);
                    assertEquals(header, string(z, "labelV3"), id);
                    assertEquals(Double.parseDouble(expected.get(header)), value(z), TOLERANCE, id + " " + header);
                    assertTrue(value(p) >= 0 && value(p) <= 100, id + " " + percentile);
                    checked += 2;
                }
                assertEquals(16, checked, id + " components in " + scoresFile);
                System.out.println(id + ": 16 CLEAR Text Easability records match the PCA, " + selectedLabels
                        + " alone identical");
            }
        } finally {
            composer.shutdown();
            selectedComposer.shutdown();
        }
    }

    private static DUUIComposer composer(String includeIndices) throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());
        composer.addDriver(new DUUIRemoteDriver());
        DUUIRemoteDriver.Component component = new DUUIRemoteDriver.Component(COH_METRIX_URL).withScale(1);
        if (includeIndices != null) {
            component.withParameter("include_indices", includeIndices);
        }
        composer.add(component.build().withTimeout(500L));
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

    /** All Coh-Metrix records by internal label (unique per output; several records can share a public label). */
    private static Map<String, FeatureStructure> byInternalLabel(CAS cas) {
        Type indexType = cas.getTypeSystem().getType(INDEX_TYPE);
        Map<String, FeatureStructure> records = new LinkedHashMap<>();
        for (FeatureStructure index : indexed(cas, indexType)) {
            records.put(string(index, "labelTTLab"), index);
        }
        return records;
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

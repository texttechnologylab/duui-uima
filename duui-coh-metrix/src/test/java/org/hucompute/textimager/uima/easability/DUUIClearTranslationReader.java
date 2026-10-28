package org.hucompute.textimager.uima.easability;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.MetaDataStringField;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.uima.jcas.JCas;
import org.json.JSONException;
import org.json.JSONObject;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUICollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.monitoring.AdvancedProgressMeter;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * Reads translated CLEAR excerpts ({@code translation_eval/translate_clear.py}: one JSON file per excerpt, named by
 * its ID), one CAS per excerpt, with the metadata of the original corpus.
 *
 * <p>The translation is the document text, unchanged. Its paragraphs are separated by blank lines (the excerpt's
 * paragraphs are sent that way), so each block between blank lines is one {@link Paragraph}. Translations with a
 * different number of paragraphs than the original are skipped and logged: besides merged or split paragraphs,
 * this catches answers with a preamble or translation notes, which would distort the reference corpus.
 *
 * <p>Metadata as in {@link DUUIClearCorpusReader}: all columns of {@code CLEAR_corpus_final.csv} except the English
 * excerpt as {@link MetaDataStringField}s, plus {@code translation_*} fields (file, model, finish reason). Only translations that ended normally (finish reason {@code stop}) are read; truncated ones are skipped
 * and logged. The translation script writes each file atomically, so the reader can be used while it is running
 * (it reads the files present when it is created).
 */
public class DUUIClearTranslationReader implements DUUICollectionReader {

    public static final String COLLECTION_ID = "CLEAR translation";

    private static final String ID_COLUMN = "ID";
    private static final String TITLE_COLUMN = "Title";
    private static final String TEXT_COLUMN = "Excerpt";
    private static final String FINISH_REASON_STOP = "stop";
    private static final Pattern PARAGRAPH = Pattern.compile("[^\\n]+(?:\\n[ \\t]*[^\\s][^\\n]*)*");

    private record Translation(Path file, JSONObject json) {
    }

    private final List<String> header;
    private final Map<String, CSVRecord> records = new HashMap<>();
    private final List<Translation> translations = new ArrayList<>();
    private final String language;
    private final AdvancedProgressMeter progress;
    private final AtomicInteger next = new AtomicInteger(0);

    public DUUIClearTranslationReader(Path translationDirectory, Path csvFile, String language) throws IOException {
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setDelimiter('\t')
                .setHeader()
                .setSkipHeaderRecord(true)
                .build();
        try (Reader in = Files.newBufferedReader(csvFile, StandardCharsets.UTF_8);
             CSVParser parser = format.parse(in)) {
            header = parser.getHeaderNames();
            for (CSVRecord record : parser.getRecords()) {
                records.put(record.get(ID_COLUMN), record);
            }
        }

        List<Path> files;
        try (Stream<Path> list = Files.list(translationDirectory)) {
            files = list.filter(path -> path.getFileName().toString().endsWith(".json")).sorted().toList();
        }
        for (Path file : files) {
            JSONObject json;
            try {
                json = new JSONObject(Files.readString(file, StandardCharsets.UTF_8));
            } catch (JSONException e) {
                System.err.println("Skipping " + file + ": not valid JSON (still being written?): " + e.getMessage());
                continue;
            }
            String finishReason = finishReason(json);
            if (!FINISH_REASON_STOP.equals(finishReason)) {
                System.err.println("Skipping " + file + ": finish reason " + finishReason);
                continue;
            }
            // e.g. a preamble ("Hier ist eine ... Übersetzung:") and notes after "---" (1146 of kolibri-1-q4km)
            if (json.getInt("paragraphs_source") != json.getInt("paragraphs_translation")) {
                System.err.println("Skipping " + file + ": " + json.getInt("paragraphs_source") + " -> "
                        + json.getInt("paragraphs_translation") + " paragraphs");
                continue;
            }
            if (!records.containsKey(json.getString("id"))) {
                throw new IllegalArgumentException(file + ": ID " + json.getString("id") + " not in " + csvFile);
            }
            translations.add(new Translation(file, json));
        }
        System.out.println(translations.size() + " of " + files.size() + " translations in " + translationDirectory);

        this.language = language;
        progress = new AdvancedProgressMeter(translations.size());
    }

    private static String finishReason(JSONObject json) {
        return json.getJSONObject("response").getJSONArray("choices").getJSONObject(0).optString("finish_reason");
    }

    @Override
    public AdvancedProgressMeter getProgress() {
        return progress;
    }

    @Override
    public void getNextCas(JCas jCas) {
        int index = next.getAndIncrement();
        if (index >= translations.size()) {
            return;
        }
        Translation translation = translations.get(index);
        JSONObject json = translation.json();
        CSVRecord record = records.get(json.getString("id"));

        String text = json.getString("translation");
        jCas.setDocumentText(text);
        jCas.setDocumentLanguage(language);

        DocumentMetaData meta = DocumentMetaData.create(jCas);
        meta.setDocumentId(record.get(ID_COLUMN));
        meta.setDocumentTitle(record.get(TITLE_COLUMN));
        meta.setCollectionId(COLLECTION_ID);

        for (String column : header) {
            if (!column.equals(TEXT_COLUMN)) {
                addField(jCas, column, record.get(column));
            }
        }
        addField(jCas, "translation_file", translation.file().getFileName().toString());
        addField(jCas, "translation_model", json.getJSONObject("response").optString("model"));
        addField(jCas, "translation_finish_reason", finishReason(json));

        // one paragraph per block between blank lines, without surrounding whitespace
        Matcher paragraph = PARAGRAPH.matcher(text);
        while (paragraph.find()) {
            int begin = paragraph.start();
            int end = paragraph.end();
            while (begin < end && Character.isWhitespace(text.charAt(begin))) {
                begin++;
            }
            while (end > begin && Character.isWhitespace(text.charAt(end - 1))) {
                end--;
            }
            if (begin < end) {
                new Paragraph(jCas, begin, end).addToIndexes();
            }
        }

        progress.next();
    }

    private static void addField(JCas jCas, String key, String value) {
        MetaDataStringField field = new MetaDataStringField(jCas);
        field.setKey(key);
        field.setValue(value);
        field.addToIndexes();
    }

    @Override
    public boolean hasNext() {
        return next.get() < translations.size();
    }

    @Override
    public long getSize() {
        return translations.size();
    }

    @Override
    public long getDone() {
        return Math.min(next.get(), translations.size());
    }
}

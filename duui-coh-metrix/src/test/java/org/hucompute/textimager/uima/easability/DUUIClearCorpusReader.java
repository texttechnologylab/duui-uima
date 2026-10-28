package org.hucompute.textimager.uima.easability;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.MetaDataStringField;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.uima.jcas.JCas;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUICollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.monitoring.AdvancedProgressMeter;

import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Reads the CommonLit Ease of Readability (CLEAR) corpus (Crossley et al. 2022,
 * https://github.com/scrosseye/CLEAR-Corpus) from {@code CLEAR_corpus_final.csv}, one CAS per excerpt.
 *
 * <p>Despite its extension the file is tab-separated; excerpts are quoted and contain line breaks. The excerpt is
 * the document text, unchanged. Each line of the excerpt is one paragraph (the line count matches the corpus'
 * {@code Paragraphs} column for all 4724 excerpts), so {@link Paragraph} annotations are added here.
 *
 * <p>All other columns are stored as {@link MetaDataStringField}s keyed by the column name (see
 * {@code CLEAR_corpus_final_keys.csv}); the document id is the corpus' {@code ID}.
 */
public class DUUIClearCorpusReader implements DUUICollectionReader {

    public static final String COLLECTION_ID = "CLEAR";

    private static final String ID_COLUMN = "ID";
    private static final String TITLE_COLUMN = "Title";
    private static final String TEXT_COLUMN = "Excerpt";

    private final List<String> header;
    private final List<CSVRecord> records;
    private final String language;
    private final AdvancedProgressMeter progress;
    private final AtomicInteger next = new AtomicInteger(0);

    public DUUIClearCorpusReader(Path csvFile, String language) throws IOException {
        CSVFormat format = CSVFormat.DEFAULT.builder()
                .setDelimiter('\t')
                .setHeader()
                .setSkipHeaderRecord(true)
                .build();
        try (Reader in = Files.newBufferedReader(csvFile, StandardCharsets.UTF_8);
             CSVParser parser = format.parse(in)) {
            header = parser.getHeaderNames();
            records = parser.getRecords();
        }
        this.language = language;
        progress = new AdvancedProgressMeter(records.size());
    }

    @Override
    public AdvancedProgressMeter getProgress() {
        return progress;
    }

    @Override
    public void getNextCas(JCas jCas) {
        int index = next.getAndIncrement();
        if (index >= records.size()) {
            return;
        }
        CSVRecord record = records.get(index);

        String text = record.get(TEXT_COLUMN);
        jCas.setDocumentText(text);
        jCas.setDocumentLanguage(language);

        DocumentMetaData meta = DocumentMetaData.create(jCas);
        meta.setDocumentId(record.get(ID_COLUMN));
        meta.setDocumentTitle(record.get(TITLE_COLUMN));
        meta.setCollectionId(COLLECTION_ID);

        for (String column : header) {
            if (column.equals(TEXT_COLUMN)) {
                continue;
            }
            MetaDataStringField field = new MetaDataStringField(jCas);
            field.setKey(column);
            field.setValue(record.get(column));
            field.addToIndexes();
        }

        // one paragraph per non-empty line, without surrounding whitespace
        int lineBegin = 0;
        while (lineBegin <= text.length()) {
            int lineEnd = text.indexOf('\n', lineBegin);
            if (lineEnd < 0) {
                lineEnd = text.length();
            }
            int begin = lineBegin;
            int end = lineEnd;
            while (begin < end && Character.isWhitespace(text.charAt(begin))) {
                begin++;
            }
            while (end > begin && Character.isWhitespace(text.charAt(end - 1))) {
                end--;
            }
            if (begin < end) {
                new Paragraph(jCas, begin, end).addToIndexes();
            }
            lineBegin = lineEnd + 1;
        }

        progress.next();
    }

    @Override
    public boolean hasNext() {
        return next.get() < records.size();
    }

    @Override
    public long getSize() {
        return records.size();
    }

    @Override
    public long getDone() {
        return Math.min(next.get(), records.size());
    }
}

import org.apache.uima.UIMAException;
import org.apache.uima.collection.CollectionReader;
import org.apache.uima.fit.factory.AggregateBuilder;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.pipeline.SimplePipeline;
import org.dkpro.core.io.xmi.XmiWriter;
import org.texttechnologylab.duui.io.hocr.HOCRReader;

import java.io.IOException;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class HOCRTest {
    static public void main(String[] args) throws UIMAException, IOException {

        CollectionReader reader = CollectionReaderFactory.createReader(
                HOCRReader.class
                , HOCRReader.PARAM_SOURCE_LOCATION, "src/test/resources/hocr"
                , HOCRReader.PARAM_PATTERNS,"*.html"
                , HOCRReader.PARAM_DOC_MODIFICATION_USER, "tester"
                , HOCRReader.PARAM_DOC_MODIFICATION_COMMENT, "testing"
                , HOCRReader.PARAM_LANGUAGE, "de"
        );

        AggregateBuilder builder = new AggregateBuilder();

        builder.add(createEngineDescription(
                XmiWriter.class
                , XmiWriter.PARAM_TARGET_LOCATION, "src/test/resources/hocr_out"
                , XmiWriter.PARAM_OVERWRITE, true
        ));

        SimplePipeline.runPipeline(reader, builder.createAggregate());
    }
}

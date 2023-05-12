package org.texttechnologylab.duui.io.json;

import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.json.JsonCasSerializer;
import org.dkpro.core.api.io.JCasFileWriter_ImplBase;

import java.io.IOException;
import java.io.OutputStreamWriter;

/**
 * JSONWriter
 * <p>
 * Writes the data into a JSON file
 *
 * @author Anna-Lena Buccoli
 */
public class JsonWriter extends JCasFileWriter_ImplBase {

    /**
     * Parameter defines the output format
     */
    public static final String PARAM_PRETTY_PRINT = "prettyPrint";
    @ConfigurationParameter(name = PARAM_PRETTY_PRINT, mandatory = false, defaultValue = "true")
    private boolean prettyPrint;

    /**
     * Parameter defines which context format is used
     */
    public static final String PARAM_JSON_CONTEXT_FORMAT = "jsonContextFormat";
    @ConfigurationParameter(name = PARAM_JSON_CONTEXT_FORMAT, mandatory = false, defaultValue = "omitExpandedTypeNames")
    private String jsonContextFormat;

    /**
     * Converts the given JCas into a JSON file
     *
     * @param aJCas contains the data
     * @throws AnalysisEngineProcessException
     */
    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {

        try {
            OutputStreamWriter out = new OutputStreamWriter(getOutputStream(aJCas, ".org.texttechnologylab.duui.io.json"));

            // Serialize data
            JsonCasSerializer serializer = new JsonCasSerializer();
            serializer.setPrettyPrint(prettyPrint);
            serializer.setJsonContext(JsonCasSerializer.JsonContextFormat.valueOf(jsonContextFormat));

            serializer.serialize(aJCas.getCas(), out);

        } catch (IOException e) {
            throw new AnalysisEngineProcessException();
        }
    }
}

import org.apache.uima.UimaContext;
import org.apache.uima.analysis_engine.AnalysisEngineProcessException;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.XMLSerializer;
import org.dkpro.core.api.io.JCasFileWriter_ImplBase;
import org.dkpro.core.api.parameter.ComponentParameters;

import javax.xml.transform.OutputKeys;
import java.io.OutputStream;
import java.io.StringWriter;
import java.nio.charset.Charset;

public class PlainWriter extends JCasFileWriter_ImplBase {

    public static final String PARAM_FILENAME_EXTENSION =
            ComponentParameters.PARAM_FILENAME_EXTENSION;
    @ConfigurationParameter(name = PARAM_FILENAME_EXTENSION, mandatory = true, defaultValue = ".xmi")
    private String filenameSuffix;

    @Override
    public void initialize(UimaContext aContext)
            throws ResourceInitializationException {
        super.initialize(aContext);
    }

    @Override
    public void process(JCas aJCas) throws AnalysisEngineProcessException {

        try (OutputStream docOS = getOutputStream(aJCas, filenameSuffix)) {

            docOS.write(aJCas.getDocumentText().getBytes(Charset.forName("UTF-8")));

        }
        catch (Exception e) {
            throw new AnalysisEngineProcessException(e);
        }

    }

}

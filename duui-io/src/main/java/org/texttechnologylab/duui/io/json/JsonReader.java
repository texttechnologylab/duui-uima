package org.texttechnologylab.duui.io.json;

import org.apache.commons.io.IOUtils;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.api.io.JCasResourceCollectionReader_ImplBase;
import org.dkpro.core.api.parameter.ComponentParameters;
import org.json.JSONObject;

import java.io.IOException;

/**
 * JSONReader
 * <p>
 * Reads the data from a JSON file
 *
 * @author Anna-Lena Buccoli
 */
public class JsonReader extends JCasResourceCollectionReader_ImplBase {

    /**
     * Parameter defines the encoding
     */
    public static final String PARAM_SOURCE_ENCODING = "sourceEncoding";
    @ConfigurationParameter(name = PARAM_SOURCE_ENCODING, mandatory = false, defaultValue = ComponentParameters.DEFAULT_ENCODING)
    private String sourceEncoding;

    /**
     * Reads a JSON file and convert it into a CAS
     *
     * @param jCas contains the data
     * @throws IOException
     */
    @Override
    public void getNext(JCas jCas) throws IOException {
        Resource res = nextFile();
        initCas(jCas, res);

        JSONObject jsonObject = new JSONObject(IOUtils.toString(res.getInputStream(), sourceEncoding));

        // Set language and text into the CAS Object
        String docLanguage = jsonObject.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0).getString("language");
        String sofa = jsonObject.getJSONObject("_views").getJSONObject("_InitialView").getJSONArray("DocumentMetaData").getJSONObject(0).get("sofa").toString();
        String docText = jsonObject.getJSONObject("_referenced_fss").getJSONObject(sofa).getString("sofaString");

        jCas.setDocumentLanguage(docLanguage);
        jCas.setDocumentText(docText);

    }
}

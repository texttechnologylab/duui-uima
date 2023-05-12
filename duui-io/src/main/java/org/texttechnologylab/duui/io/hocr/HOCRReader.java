package org.texttechnologylab.duui.io.hocr;

import org.apache.uima.UIMAException;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.api.io.JCasResourceCollectionReader_ImplBase;
import org.dkpro.core.api.resources.CompressionUtils;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

public class HOCRReader extends JCasResourceCollectionReader_ImplBase {
    @Override
    public void getNext(JCas jCas) throws IOException, CollectionException {
        Resource res = nextFile();
        initCas(jCas, res);
        try (InputStream is = new BufferedInputStream(CompressionUtils.getInputStream(res.getLocation(), res.getInputStream()))) {
            HOCRLoader.load(is, jCas);
        } catch (ParserConfigurationException | UIMAException | SAXException e) {
            throw new CollectionException(e);
        }
    }
}

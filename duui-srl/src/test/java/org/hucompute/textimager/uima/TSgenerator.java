package org.hucompute.textimager.uima;

import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.TypeSystemUtil;
import org.junit.jupiter.api.Test;
import org.xml.sax.SAXException;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class TSgenerator {
    @Test
    public void TSgenerator() throws UIMAException, IOException, SAXException {
        JCas c = JCasFactory.createJCas();
        TypeSystemUtil.typeSystem2TypeSystemDescription(c.getTypeSystem()).toXML(new FileOutputStream(new File("/home/mkonca/TypeSystem.xml")));
    }
}

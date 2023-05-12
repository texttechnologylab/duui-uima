package org.hucompute.textimager.uima;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.fit.util.JCasUtil;
import com.google.common.io.Resources;
import org.apache.commons.io.FileUtils;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasIOUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.apache.uima.util.XMLSerializer;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
//import de.unihd.dbs.uima.annotator.heideltime.biofid.HeidelTimeBioFID;
import org.xml.sax.SAXException;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;


public class CorefGerMultiTest {
    @Test
    public void test() throws Exception {

        ArrayList<String> files = new ArrayList<>();
        //files.add("bfsrl.xmi");
        files.add("1001-SZ-0210200.xmi");
        //files.add("bfsrl2.xmi");
        //files.add("test2.xmi");

        for (String filename : files) {
            System.out.println(filename);
            URL url = Resources.getResource(filename);
            JCas jcas1 = JCasFactory.createJCas();
            CasIOUtil.readXmi(jcas1.getCas(), new File(url.toURI()));
            //CasIOUtils.load(url, jcas1.getCas());
            //System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
            jcas1.setDocumentLanguage("de");

            for (Sentence s : JCasUtil.select(jcas1, Sentence.class)){
                System.out.println(":)");
                for (Token t : JCasUtil.selectCovered(Token.class, s)){
                    System.out.println(t.getCoveredText());
                }
            }
            //System.exit(0);
            DUUIComposer composer = new DUUIComposer()
                    .withSkipVerification(true)
                    .withLuaContext(new DUUILuaContext().withJsonLibrary());

            DUUIRemoteDriver remote_driver = new DUUIRemoteDriver();
            composer.addDriver(remote_driver);

            composer.add(
                    new DUUIRemoteDriver.Component("http://127.0.0.1:8000")
                            .withParameter("model_name", "dkp")
            );

            composer.run(jcas1);
            System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
            composer.shutdown();
        }
    }
}

package org.hucompute.textimager.uima.transformers.summary;

import com.google.common.io.Resources;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasIOUtil;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;
import org.apache.uima.jcas.tcas.Annotation;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import org.texttechnologylab.annotation.Summary;

public class GermanSummaryTest {
    static DUUIComposer composer;
    static String summery_url = "http://127.0.0.1:8000";
    static String model = "Google T5";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }


    public JCas createCas(String language, List<String> sentences) throws UIMAException {
        JCas cas = JCasFactory.createJCas();
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length()+sentence.length());
            sentenceAnnotation.addToIndexes();
            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());

        return cas;
    }

//    @Test
//    public void documentTest() throws Exception {
//        URL url = Resources.getResource("1001-SZ-02102004.tei119.xmi");
//        JCas jcas1 = JCasFactory.createJCas();
//        //CasIOUtil.readXmi(jcas1.getCas(), new File(url.toURI()));
//        CasIOUtils.load(url, jcas1.getCas());
//        System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
//        jcas1.setDocumentLanguage("de");
//
//        composer.add(
//                new DUUIRemoteDriver.Component(summery_url)
//                        .withParameter("model_name", model)
//                        .withParameter("summary_length", "75")
//        );
//
//        composer.run(jcas1);
//        System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
//    }

    @Test
    public void sentencesTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(summery_url)
                        .withParameter("model_name", model)
                        .withParameter("summary_length", "75")
        );

        String Text = "Der neueste Streich aus dem Hause Open AI hilft auch in der Küche. Das neue große Sprachmodell GPT-4 des amerikanischen Unternehmens ist nämlich multimodal, es kann also auch Bilder verarbeiten. So erkennt die Künstliche Intelligenz (KI) mit Hilfe eines Bildes aus dem Inneren eines Kühlschranks nicht nur, welche Inhalte sich darin befinden, sondern gibt auch Vorschläge, was damit gekocht werden könnte. Es ist nur eine von vielen Spielereien, die die neuste Version des KI-Modells ermöglicht, auf dem der Dienst ChatGPT basiert. Es ist zuverlässiger und kreativer als sein Vorgänger, obwohl es immer noch anfällig für falsche Antworten ist und auch nicht aktuell ist. Das Modell basiert auf Daten aus dem Jahr 2021. Dennoch zeigt GPT-4, was für enorme Fortschritte im Bereich der sogenannten generativen KI in den vergangenen Jahren erzielt wurden.";
        JCas cas = JCasFactory.createText(Text, "de");

        Annotation sentence1 = new Sentence(cas, 0, 405);
        sentence1.addToIndexes();
        Annotation sentence2 = new Sentence(cas, 406, Text.length());
        sentence2.addToIndexes();

        composer.run(cas);

        composer.shutdown();

        Collection<Summary> summaries = JCasUtil.select(cas, Summary.class);

        for (Summary summary : summaries) {
            System.out.println(summary.getSummary());
        }
        assert summaries.size() == 3;



        System.out.println(XmlFormatter.getPrettyString(cas.getCas()));

    }
}

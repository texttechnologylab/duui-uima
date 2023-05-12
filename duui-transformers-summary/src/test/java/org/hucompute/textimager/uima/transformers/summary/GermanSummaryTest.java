package org.hucompute.textimager.uima.transformers.summary;

import com.google.common.io.Resources;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.CasIOUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

public class GermanSummaryTest {
    static DUUIComposer composer;
    static String summery_url = "http://127.0.0.1:8000";
    static String model = "Google T5-base";

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

    @Test
    public void documentTest() throws Exception {
        URL url = Resources.getResource("1001-SZ-02102004.tei119.xmi");
        JCas jcas1 = JCasFactory.createJCas();
        //CasIOUtil.readXmi(jcas1.getCas(), new File(url.toURI()));
        CasIOUtils.load(url, jcas1.getCas());
        System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
        jcas1.setDocumentLanguage("de");

        composer.add(
                new DUUIRemoteDriver.Component(summery_url)
                        .withParameter("model_name", model)
                        .withParameter("summary_length", "75")
        );

        composer.run(jcas1);
        System.out.println(XmlFormatter.getPrettyString(jcas1.getCas()));
    }

    @Test
    public void sentencesTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(summery_url)
                        .withParameter("model_name", model)
                        .withParameter("summary_length", "75")
        );

        List<String> sentences = Arrays.asList(
                "Der neueste Streich aus dem Hause Open AI hilft auch in der Küche.",
                "Das neue große Sprachmodell GPT-4 des amerikanischen Unternehmens ist nämlich multimodal, es kann also auch Bilder verarbeiten.",
                "So erkennt die Künstliche Intelligenz (KI) mit Hilfe eines Bildes aus dem Inneren eines Kühlschranks nicht nur, welche Inhalte sich darin befinden, sondern gibt auch Vorschläge, was damit gekocht werden könnte.",
                "Es ist nur eine von vielen Spielereien, die die neuste Version des KI-Modells ermöglicht, auf dem der Dienst ChatGPT basiert.",
                "Es ist zuverlässiger und kreativer als sein Vorgänger, obwohl es immer noch anfällig für falsche Antworten ist und auch nicht aktuell ist.",
                "Das Modell basiert auf Daten aus dem Jahr 2021.",
                "Dennoch zeigt GPT-4, was für enorme Fortschritte im Bereich der sogenannten generativen KI in den vergangenen Jahren erzielt wurden.",
                "Die KI besteht ein juristisches Staatsexamen – und ist dabei laut Open-AI unter den besten 10 Prozent.",
                "Das Modell GPT-3.5, auf dem aktuell ChatGPT basiert, schnitt noch unter den letzten 10 Prozent ab.",
                "Die Technologie basiert auf sogenannten tiefen neuronalen Netzen.",
                "Dieses tiefe Lernen ist eine noch anspruchsvollere Version des maschinellen Lernens, bei der spezielle Algorithmen mit riesigen Datenmengen darauf trainiert werden, bestimmte Zusammenhänge zu erkennen und später auch auf unbekannte Daten anwenden zu können.",
                "Der aktuelle Fortschritt hängt auch damit zusammen, dass die sogenannten großen KI-Modelle mit der Zeit immer größer geworden sind.",
                "Ihre Größe wird unter anderem anhand ihrer Parameter bestimmt.",
                "Das sind die Werte, welche die KI während ihres Trainings lernt und durch weiteres Training präzisiert.",
                "Neben dem Umfang der Daten und dem Rechenaufwand für das Training hat die Anzahl der Parameter einen maßgeblichen Einfluss auf die Leistungsfähigkeit der Modelle.",
                "Den enormen Sprung zeigt etwa der Vergleich der Parameter des 2019 veröffentlichten Modells GPT-2 mit seinem Nachfolger GPT-3.",
                "So enthielt GPT-2 noch 1,5 Milliarden Parameter und wurde mit 40 Gigabyte Text trainiert.",
                "Das künstliche neuronale Netz im Kern von GPT-3 enthält mehr als 100-mal so viele Parameter, nämlich stolze 175 Milliarden, und wurde mit 2 Terabyte Text gefüttert.",
                "Wie viele Parameter in GPT-4 stecken, ist nicht bekannt.",
                "Das neuste KI-Modell des Internetkonzerns Google enthält aber sogar 540 Milliarden Parameter.",
                "Auch ein weiterer Indikator weist auf die rasanten technischen Fortschritte in der Technologie hin.",
                "Die Anzahl der Patentanmeldungen im Bereich KI stieg von knapp 81.000 im Jahr 2020 auf mehr als 140.000 im Jahr 2021.",
                "2016 waren es erst 6000 gewesen.");

        JCas cas = createCas("de", sentences);
        composer.run(cas);

        System.out.println(XmlFormatter.getPrettyString(cas.getCas()));

    }
}

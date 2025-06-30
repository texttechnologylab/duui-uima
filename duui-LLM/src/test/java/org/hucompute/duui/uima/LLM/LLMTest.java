package org.hucompute.duui.uima.LLM;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.AnnotationBase;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.jcas.cas.TOP;
import org.apache.uima.jcas.tcas.Annotation;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.*;
import org.texttechnologylab.type.LLMPrefixPrompt;
import org.texttechnologylab.type.LLMPrompt;
import org.texttechnologylab.type.LLMSuffixPrompt;
import org.texttechnologylab.type.LLMResult;
import org.texttechnologylab.type.LLMSystemPrompt;

public class LLMTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException, CompressorException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
//        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
//        composer.addDriver(docker_driver);


        cas = JCasFactory.createJCas();
    }

    @AfterAll
    static void afterAll() throws UnknownHostException {
        composer.shutdown();
    }

    @AfterEach
    public void afterEach() throws IOException, SAXException {
        composer.resetPipeline();

        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        XmlCasSerializer.serialize(cas.getCas(), null, stream);
        System.out.println(stream.toString(StandardCharsets.UTF_8));

        cas.reset();
    }

    @Test
    public void EnglishTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("seed", "42")
                        .withParameter("model_name", "CORE:Keyword-ExtractionV2_Gemma-2-27B_v1")
                        .withParameter("url", "anduin.hucompute.org")
                        .withParameter("temperature", "1")
                        .withParameter("port", "11434")
        );
        String prompt = "";
        String suffix = "\n \nGebe Ausschließlich die Schlagwörter als JSON aus keinen weiteren Text.";
        String task = "Schreiben Sie eine kurze, prägnante Stellungnahme, in der Sie die zentralen ökonomischen Faktoren nennen, welche die Wirkung von o.g. Nudging-Maßnahmen auf Stromsparen begünstigen oder beeinträchtigen könnten. Belegen Sie Ihre Position mit mindestens 2 der gefundenen Quellen (ca. 10 Minuten).\n" +
                "\n" +
                "Ihre Stellungnahme (mind. 150 bis 250 Wörter):\n";
        String scene = "Sie sind als Werkstudent:in in einer Abteilung des Wirtschaftsministeriums tätig, die sich aktuell mit der Frage beschäftigt, wie man den Verbrauch elektrischer Energie weiter reduzieren kann. Ihre Abteilung erwägt als Mittel nicht nur Gesetze und Vorschriften, deren Einhaltung kaum zu kontrollieren sei und die einen Eingriff in die persönliche Entscheidungsfreiheit der Individuen bedeuten können. Eine Alternative wird in dem sogenannten ,Nudging' (dt. ,anstoßen') gesehen. Dabei sollen Menschen etwa durch das Setzen geeigneter technischer Standardeinstellungen zu stromsparendem Verhalten angeregt werden.\n" +
                "\n" +
                "Eine Idee besteht darin, elektrische Geräte mit intelligenten Anzeigesystemen und Sensoren auszustatten, die den Stromverbrauch und den täglichen Verlauf der Stromverbrauchskurve auf dem Handy anzeigen. Dies könnte künftig (1) täglich beim Anschalten des Handys durch eine feste Handy-Voreinstellung oder (2) auf Abfrage, an die die Nutzer:innen täglich mit einer Push-Benachrichtigung erinnert werden, erfolgen. Die Handy-Hersteller würden dabei je nach Regierungsbeschluss verpflichtet, die Version (1) oder (2) zu implementieren.\n" +
                "\n" +
                "Sie erhalten den Auftrag, ein Expose für eine wirtschaftspolitische Handlungsempfehlung zur Regulierung des elektrischen Verbrauchs durch Nudging vorzubereiten. Die Abteilungsleiterin erwartet Ihre Rechercheergebnisse im Verlauf der nächsten Stunde, da die Abteilungssitzung kurz bevorsteht. Aus diesen Zeitgründen sollen Sie sich auf die folgenden zwei Arbeitsaufträge fokussieren, für die Sie jeweils ca. 25 Minuten einplanen\n";
        prompt = "Szenario:\n" + scene + "\n\n" +
                "Aufgabenstellung:\n" + task;
        StringBuilder sb = new StringBuilder();
        JCas cas = JCasFactory.createJCas();
        LLMSuffixPrompt suffixPrompt = new LLMSuffixPrompt(cas, sb.length(), sb.length() + suffix.length());
        suffixPrompt.setMessage(suffix);
        suffixPrompt.addToIndexes();
        sb.append(suffix).append(" ");
//        LLMSystemPrompt systemPromptObj = new LLMSystemPrompt(cas, sb.length(), sb.length() + systemPrompt.length());
//        systemPromptObj.setMessage(systemPrompt);
//        systemPromptObj.addToIndexes();
//        sb.append(systemPrompt).append(" ");
        LLMPrompt llmPrompt = new LLMPrompt(cas, sb.length(), sb.length() + prompt.length());
        llmPrompt.setPrompt(prompt);
        llmPrompt.setSuffix(suffixPrompt);
//        llmPrompt.setSystemPrompt(systemPromptObj);
        llmPrompt.addToIndexes();
        sb.append(prompt).append(" ");
        cas.setDocumentText(sb.toString());
        cas.setDocumentLanguage("en");
        LLMPrompt llmPromptObj = JCasUtil.selectSingle(cas, LLMPrompt.class);
        composer.run(cas);

        Collection<LLMResult> llmResults = JCasUtil.select(cas, LLMResult.class);
        for (LLMResult llmResult : llmResults) {
            System.out.println("LLM Result: " + llmResult.getContent());
            // Check type of the result
            Assertions.assertEquals("String", llmResult.getContent().getClass().getSimpleName());
            // Check if the result is not empty
            Assertions.assertFalse(llmResult.getContent().isEmpty());
        }

        System.out.println("Run Composer");


    }

}

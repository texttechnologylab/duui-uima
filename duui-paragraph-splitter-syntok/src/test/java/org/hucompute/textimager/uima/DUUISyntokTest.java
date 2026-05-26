package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DUUISyntokTest {
    static DUUIComposer composer;
    static JCas cas;

    static String dockerImage = "docker.texttechnologylab.org/duui-syntok:0.0.2";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

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

    public void createCas(String language, List<String> sentences) {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        if (sentences != null) {
            for (String sentence : sentences) {
                sb.append(sentence).append(" ");
            }
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void sentencesTest() throws Exception {
        composer.add(
                new DUUIDockerDriver.Component(dockerImage)
//                new DUUIRemoteDriver.Component("http://localhost:8000")
                        .withParameter("write_sentences", String.valueOf(true))  // default false
                        .withParameter("write_paragraphs", String.valueOf(true)) // default true
        );

        List<String> expectedSentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example.",
                "I absolutely hate this example 2.\n\nI absolutely hate this example 3."
        );

        createCas("en", expectedSentences);
        composer.run(cas);

        for (Paragraph paragraph : JCasUtil.select(cas, Paragraph.class)) {
            System.out.println("Paragraph begin: " + paragraph.getBegin());
            System.out.println("Paragraph end: " + paragraph.getEnd());
            System.out.println("Paragraph: !" + paragraph.getCoveredText() + "!");
            for (Sentence sentence : JCasUtil.selectCovered(Sentence.class, paragraph)) {
                System.out.println("  Sentence begin: " + sentence.getBegin());
                System.out.println("  Sentence end: " + sentence.getEnd());
                System.out.println("  Sentence: !" + sentence.getCoveredText() + "!");
            }
        }
    }
}

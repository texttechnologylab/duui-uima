package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.AnnotatorMetaData;
import org.texttechnologylab.annotation.DocumentModification;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class DUUISentencizerSegtokTest {
    static DUUIComposer composer;
    static JCas cas;

    static String dockerImage = "docker.texttechnologylab.org/duui-sentencizer-segtok:latest";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException, SAXException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

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
                        .withImageFetching()
        );

        List<String> expectedSentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example."
        );
        Integer[] expectedSentenceSpansBegin = new Integer[]{ 0, 0, 39 };
        Integer[] expectedSentenceSpansEnd = new Integer[]{ 38, 0, 70 };

        createCas("en", expectedSentences);
        composer.run(cas);

        Collection<AnnotatorMetaData> actualAnnotatorMetaDatas = new ArrayList<>(JCasUtil.select(cas, AnnotatorMetaData.class));
        assertEquals(3, actualAnnotatorMetaDatas.size());

        Collection<DocumentModification> actualDocumentModifications = new ArrayList<>(JCasUtil.select(cas, DocumentModification.class));
        assertEquals(1, actualDocumentModifications.size());

        Collection<Sentence> actualSentences = new ArrayList<>(JCasUtil.select(cas, Sentence.class));
        assertEquals(expectedSentences.size()+1, actualSentences.size());

        Integer[] actualSenttencesSpansBegin = actualSentences.stream().map(Sentence::getBegin).toArray(Integer[]::new);
        assertArrayEquals(expectedSentenceSpansBegin, actualSenttencesSpansBegin);

        Integer[] actualSenttencesSpansEnd = actualSentences.stream().map(Sentence::getEnd).toArray(Integer[]::new);
        assertArrayEquals(expectedSentenceSpansEnd, actualSenttencesSpansEnd);
    }

    @Test
    public void emptyTest() throws Exception {
        composer.add(
                new DUUIDockerDriver.Component(dockerImage)
                        .withImageFetching()
        );

        createCas("en", null);
        composer.run(cas);

        Collection<AnnotatorMetaData> actualAnnotatorMetaDatas = new ArrayList<>(JCasUtil.select(cas, AnnotatorMetaData.class));
        assertEquals(1, actualAnnotatorMetaDatas.size());

        Collection<DocumentModification> actualDocumentModifications = new ArrayList<>(JCasUtil.select(cas, DocumentModification.class));
        assertEquals(1, actualDocumentModifications.size());

        Collection<Sentence> actualSentences = new ArrayList<>(JCasUtil.select(cas, Sentence.class));
        assertEquals(1, actualSentences.size());
    }
}
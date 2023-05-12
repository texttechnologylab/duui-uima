package org.hucompute.textimager.uima.ddc.fasttext;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Lemma;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.XmlCasSerializer;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

public class DDCFastTextTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:9714";

    @BeforeAll
    static void beforeAll() throws URISyntaxException, IOException, UIMAException {
        composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length()+sentence.length());
            sentenceAnnotation.addToIndexes();

            // TODO use only for testing!
            int tokenInd = 0;
            String[] simpleTokens = sentence.split(" ");
            for (String token : simpleTokens) {
                Lemma lemmaAnnotation = new Lemma(cas, tokenInd, tokenInd + token.length());
                lemmaAnnotation.setValue(token);
                lemmaAnnotation.addToIndexes();
                Token tokenAnnotation = new Token(cas, tokenInd, tokenInd + token.length());
                tokenAnnotation.setLemma(lemmaAnnotation);
                tokenAnnotation.addToIndexes();
                tokenInd += token.length() + 1;
            }

            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void sentencesDeTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("ddc_variant", "ddc2")
                        .withParameter("selection", "sentence")
        );

        List<String> sentences = Arrays.asList(
                "Dies ist ein sähr güter ßatz!",
                "Ich haße döeses Bäispiel."
        );

        createCas("de", sentences);
        composer.run(cas);

        Collection<CategoryCoveredTagged> categories = JCasUtil.select(cas, CategoryCoveredTagged.class);
        System.out.println(categories.size());
    }

    @Test
    public void sentencesDdc3DeTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("ddc_variant", "ddc3")
                        .withParameter("selection", "sentence")
        );

        List<String> sentences = Arrays.asList(
                "Dies ist ein sähr güter ßatz!",
                "Ich haße döeses Bäispiel."
        );

        createCas("de", sentences);
        composer.run(cas);

        Collection<CategoryCoveredTagged> categories = JCasUtil.select(cas, CategoryCoveredTagged.class);
        System.out.println(categories.size());
    }

    @Test
    public void sentencesEnTest() throws Exception {
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("ddc_variant", "ddc2")
                        .withParameter("selection", "text")
        );

        List<String> sentences = Arrays.asList(
                "This is a very great example sentence!",
                "I absolutely hate this example."
        );

        createCas("en", sentences);
        composer.run(cas);

        Collection<CategoryCoveredTagged> categories = JCasUtil.select(cas, CategoryCoveredTagged.class);
        System.out.println(categories.size());
    }
}


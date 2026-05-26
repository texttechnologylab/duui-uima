package org.hucompute.textSearchReference;
import de.tudarmstadt.ukp.dkpro.core.api.syntax.type.constituent.S;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
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

import org.texttechnologylab.type.LLMPrompt;
import org.texttechnologylab.type.LLMSuffixPrompt;
import org.texttechnologylab.type.LLMResult;
import org.texttechnologylab.type.search.TextSearch;
import org.texttechnologylab.annotation.search.ReferenceText;

public class TextSearchReferenceTest {
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
                        .withParameter("search_language", "de")
                        .withParameter("method", "Google")
                        .withParameter("search", "Wikipedia")
        );
        String search_text = "Nudging";
        String search_text2 = "Stromsparen";

        StringBuilder sb = new StringBuilder();
        JCas cas = JCasFactory.createJCas();
        TextSearch textSearch = new TextSearch(cas, 0, search_text.length());
        textSearch.setText(search_text);
        textSearch.addToIndexes();
        sb.append(search_text).append(" ");
        TextSearch textSearch2 = new TextSearch(cas, sb.length(), sb.length() + search_text2.length());
        textSearch2.setText(search_text2);
        textSearch2.addToIndexes();
        sb.append(search_text2).append(" ");
        cas.setDocumentText(sb.toString());
        composer.run(cas);


        System.out.println("Run Composer");
        Collection<ReferenceText> referenceTexts = JCasUtil.select(cas, ReferenceText.class);
        for (ReferenceText referenceText : referenceTexts) {
            System.out.println("ReferenceText: " + referenceText.getReference().getText());
            System.out.println("ReferenceText: " + referenceText.getBegin() + " - " + referenceText.getEnd());
            System.out.println("ReferenceText: " + referenceText.getPriority());
            System.out.println("ReferenceText: " + referenceText.getUrl());
        }


    }

}

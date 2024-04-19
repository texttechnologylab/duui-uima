package org.hucompute.textimager.uima.entailment;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.Argument;
import org.xml.sax.SAXException;
import org.apache.uima.jcas.tcas.Annotation;
import org.texttechnologylab.annotation.EntailmentSentence;
import org.texttechnologylab.annotation.EntailmentGPT;
import org.texttechnologylab.annotation.Entailment;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class EntailmentTest {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
    //    static String model = "chkla/parlbert-topic-german";
    static String model = "UKP";

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
//        System.out.println(stream.toString(StandardCharsets.UTF_8));

        cas.reset();
    }

    @Test
    public void MultlingualTest() throws Exception {
        JCas cas = JCasFactory.createText("A fossil fuel is a kind of natural resource. Coal is a kind of fossil fuel. Coal is not a kind of natural resource. Coal is a kind of natural resource.");
        cas.setDocumentLanguage("en");
        Annotation a1 = new Annotation(cas, 0, 75);
        a1.addToIndexes();
        Annotation a2 = new Annotation(cas, 76, 115);
        a2.addToIndexes();
        Annotation a3 =  new Annotation(cas, 116, 151);
        a3.addToIndexes();
        EntailmentSentence en1 = new EntailmentSentence(cas);
        en1.setPremise(a1);
        en1.setHypothesis(a2);
        en1.addToIndexes();
        EntailmentSentence en2 =  new EntailmentSentence(cas);
        en2.setHypothesis(a3);
        en2.setPremise(a1);
        en2.addToIndexes();

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("chatgpt_key", "")
        );

        composer.run(cas);
        composer.shutdown();

        Collection<Entailment> entailment_out = JCasUtil.select(cas, Entailment.class);
        for (Entailment entail_i: entailment_out){
            System.out.println(entail_i.getReference().getHypothesis().getCoveredText());
            System.out.println(entail_i.getEntailment());
            System.out.println(entail_i.getContradiction());
            System.out.println(entail_i.getReference());
        }
        assertEquals(2, entailment_out.size());


//        Collection<Stance> stances_out = JCasUtil.select(cas, Stance.class);
//        for (Stance stance_i: stances_out) {
//            System.out.println(stance_i.getSupport());
//            System.out.println(stance_i.getOppose());
//            System.out.println(stance_i.getNeutral());
//        }
//        assertEquals(3, stances_out.size());

    }

    @Test
    public void GPTTest() throws Exception {
        JCas cas = JCasFactory.createText("A fossil fuel is a kind of natural resource. Coal is a kind of fossil fuel. Coal is not a kind of natural resource. Coal is a kind of natural resource.");
        cas.setDocumentLanguage("en");
        Annotation a1 = new Annotation(cas, 0, 75);
        a1.addToIndexes();
        Annotation a2 = new Annotation(cas, 76, 115);
        a2.addToIndexes();
        Annotation a3 = new Annotation(cas, 116, 151);
        a3.addToIndexes();
        EntailmentSentence en1 = new EntailmentSentence(cas);
        en1.setPremise(a1);
        en1.setHypothesis(a2);
        en1.addToIndexes();
        EntailmentSentence en2 = new EntailmentSentence(cas);
        en2.setHypothesis(a3);
        en2.setPremise(a1);
        en2.addToIndexes();

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("chatgpt_key", "")
        );

        composer.run(cas);
        composer.shutdown();

        Collection<EntailmentGPT> entailment_out = JCasUtil.select(cas, EntailmentGPT.class);
        for (EntailmentGPT entail_i : entailment_out) {
            System.out.println(entail_i.getReference().getHypothesis().getCoveredText());
            System.out.println(entail_i.getLabel());
            System.out.println(entail_i.getConfidence());
            System.out.println(entail_i.getReason());
            System.out.println(entail_i.getReference());
        }
        assertEquals(2, entailment_out.size());
    }
}

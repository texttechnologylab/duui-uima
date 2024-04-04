package org.hucompute.textimager.uima.stance;

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
import org.texttechnologylab.annotation.StanceSentence;
import org.texttechnologylab.annotation.Hypothesis;
import org.texttechnologylab.annotation.Stance;
import org.texttechnologylab.annotation.StanceGPT;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class StanceTest {
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

    public ArrayList<StanceSentence> createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        Integer counter = 0;
        ArrayList<StanceSentence> s = new ArrayList<StanceSentence>();
        for (String sentence : sentences) {
            Annotation annotation_i = new Annotation(cas, sb.length(), sb.length()+sentence.length());
            StanceSentence stanceSentence = new StanceSentence(cas, sb.length(), sb.length()+sentence.length());
            annotation_i.addToIndexes();
            sb.append(sentence).append(" ");
            s.add(stanceSentence);
            stanceSentence.addToIndexes();
        }

        cas.setDocumentText(sb.toString());
        return s;
    }

    @Test
    public void MultlingualTest() throws Exception {
        JCas cas = JCasFactory.createText("I Love Trump. He is the best president ever. I hate Trump. He is the worst president ever. I am neutral about Trump. He is a president. The author of this tweet {} Trump.");
        cas.setDocumentLanguage("en");
        String hypothesis = "The author of this tweet {} Trump.";
        StanceSentence stance1= new StanceSentence(cas, 0, 44);
        stance1.addToIndexes();
        StanceSentence stance2= new StanceSentence(cas, 45, 90);
        stance2.addToIndexes();
        StanceSentence stance3= new StanceSentence(cas, 91, 135);
        stance3.addToIndexes();
        Hypothesis hypo = new Hypothesis(cas, 136, 170);
        hypo.setStances(new FSArray<>(cas, 3));
        hypo.setStances(0, stance1);
        hypo.setStances(1, stance2);
        hypo.setStances(2, stance3);
        hypo.addToIndexes();

//        Collection<Language> languagesall = JCasUtil.select(cas, Language.class);
//        for (Language language_i : languagesall) {
//
//            System.out.println(language_i.getValue());
//        }
//        System.out.println("h");
        Hypothesis hypo_i = JCasUtil.selectSingle(cas, Hypothesis.class);
        FSArray<StanceSentence> stances_all = hypo_i.getStances();
        for (StanceSentence stance_i : stances_all) {
            System.out.println(stance_i.getCoveredText());
        }

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("chatgpt_key", "")
        );

        composer.run(cas);
        composer.shutdown();


        Collection<Stance> stances_out = JCasUtil.select(cas, Stance.class);
        for (Stance stance_i: stances_out) {
            System.out.println(stance_i.getSupport());
            System.out.println(stance_i.getOppose());
            System.out.println(stance_i.getNeutral());
        }
        assertEquals(3, stances_out.size());

    }

    @Test
    public void GPTTest() throws Exception {
        JCas cas = JCasFactory.createText("I Love Trump. He is the best president ever. I hate Trump. He is the worst president ever. I am neutral about Trump. He is a president. The author of this tweet {} Trump.");
        cas.setDocumentLanguage("en");
        String hypothesis = "The author of this tweet {} Trump.";
        StanceSentence stance1= new StanceSentence(cas, 0, 44);
        stance1.addToIndexes();
        StanceSentence stance2= new StanceSentence(cas, 45, 90);
        stance2.addToIndexes();
        StanceSentence stance3= new StanceSentence(cas, 91, 135);
        stance3.addToIndexes();
        Hypothesis hypo = new Hypothesis(cas, 136, 170);
        hypo.setStances(new FSArray<>(cas, 3));
        hypo.setStances(0, stance1);
        hypo.setStances(1, stance2);
        hypo.setStances(2, stance3);
        hypo.addToIndexes();

//        Collection<Language> languagesall = JCasUtil.select(cas, Language.class);
//        for (Language language_i : languagesall) {
//
//            System.out.println(language_i.getValue());
//        }
//        System.out.println("h");
        Hypothesis hypo_i = JCasUtil.selectSingle(cas, Hypothesis.class);
        FSArray<StanceSentence> stances_all = hypo_i.getStances();
        for (StanceSentence stance_i : stances_all) {
            System.out.println(stance_i.getCoveredText());
        }

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("chatgpt_key", "")
        );

        composer.run(cas);
        composer.shutdown();


        Collection<StanceGPT> stances_out = JCasUtil.select(cas, StanceGPT.class);
        for (StanceGPT stance_i: stances_out) {
            System.out.println(stance_i.getLabel());
            System.out.println(stance_i.getConfidence());
            System.out.println(stance_i.getReason());
        }
        assertEquals(3, stances_out.size());

    }
}

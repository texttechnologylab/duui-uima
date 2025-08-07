package org.hucompute.textimager.uima.essayscorer;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
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
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Div;

import org.texttechnologylab.annotation.LLMMetric;

import static org.junit.Assert.assertEquals;

public class MultiTestEssayScorer {
    static DUUIComposer composer;
    static JCas cas;

    static String url = "http://127.0.0.1:8000";
//    static String model = "pol_emo_mDeBERTa";

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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Sentence sentenceAnnotation = new Sentence(cas, sb.length(), sb.length()+sentence.length());
            sentenceAnnotation.addToIndexes();
            sb.append(sentence).append(" ");
        }


        cas.setDocumentText(sb.toString());
    }

    @Test
    public void EnglishTest() throws Exception {
//        composer.add(new DUUIDockerDriver.
//                Component("docker.texttechnologylab.org/textimager-duui-transformers-topic:0.0.1")
//                .withParameter("model_name", model)
//                .withParameter("selection", "text,de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
//                .withScale(1)
//                .withImageFetching());
        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("div_questions", "Question1,Question2")
                        .withParameter("div_answers", "Answer1,Answer2")
        );
        String question = "What is the topic of this text?";
        String answer = "The topic of this text is about the topic of the text.";
        String question2 = "What is the sentiment of this text?";
        String answer2 = "The sentiment of this text is positive.";
        List<String> sentences = Arrays.asList(
                question,
                answer,
                question2,
                answer2
        );
        createCas("en", sentences);
        ArrayList<String> ids_list = new ArrayList<>();
        ids_list.add("Question1");
        ids_list.add("Answer1");
        ids_list.add("Question2");
        ids_list.add("Answer2");
        int i = 0;
        Collection<Sentence> sentences_list = JCasUtil.select(cas, Sentence.class);
        for (Sentence sentence : sentences_list) {
            Div div = new Div(cas, sentence.getBegin(), sentence.getEnd());
            div.setId(ids_list.get(i));
            // if i is even, set the role to "question", otherwise set it to "answer"
            if (i % 2 == 0) {
                div.setDivType("question");
            } else {
                div.setDivType("answer");
            }
            div.addToIndexes();
            i++;
        }

        composer.run(cas);

        Collection<LLMMetric> all_llm_metrics = JCasUtil.select(cas, LLMMetric.class);
        for (LLMMetric llmMetric : all_llm_metrics) {
            System.out.println("LLMMetric:" + llmMetric.getBegin());
            System.out.println("LLMMetric:" + llmMetric.getEnd());
            System.out.println("LLMMetric:" + llmMetric.getValue());
            System.out.println("Key: "+ llmMetric.getKeyName());
            System.out.println("Definition:" + llmMetric.getDefinition());
            System.out.println("Model:" + llmMetric.getModel().getModelName());
        }

    }
}

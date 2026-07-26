package org.hucompute.textimager.uima.essayscorer;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.apache.uima.util.XmlCasSerializer;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.zip.GZIPInputStream;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Div;
import org.texttechnologylab.annotation.EssayScore;
import org.texttechnologylab.annotation.model.EssayScoreModel;
import org.texttechnologylab.annotation.model.EssayScoreLLM;

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

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(2000);
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
                new DUUIRemoteDriver.Component(url).build()
                        .withParameter("div_scenarios", "p_scenario,t_scenario")
                        .withParameter("div_questions", "p_reas_question,t_reas_question")
                        .withParameter("div_answers", "p_reas_answer,t_reas_answer")
                        .withParameter("seed", "42")
                        .withParameter("model_llm", "test:DeepSeek-R1")
                        .withParameter("name_model", "DeepSeek-R1:70B")
                        .withParameter("url", "gondor.hucompute.org")
                        .withParameter("temperature", "1.0")
                        .withParameter("port", "11434")
                        .withTimeout(2000)
        );
        JCas jCas = JCasFactory.createJCas();
        try (InputStream fileStream = Files.newInputStream(Paths.get("/storage/projects/CORE/data/uce/med_t0/ta_export_spacy_paragraphs/31400.xmi.gz.xmi.gz"));
             InputStream gzipStream = new GZIPInputStream(fileStream)) {
            CasIOUtils.load(gzipStream, jCas.getCas());
        }
//        String scenario = "This text is about the topic of the text. It discusses various aspects and provides insights.";
//        String question = "What is the topic of this text?";
//        String answer = "The topic of this text is about the topic of the text.";
//        String scenario2 = "This text is about sentiment analysis. It discusses how to determine the sentiment of a text.";
//        String question2 = "What is the sentiment of this text?";
//        String answer2 = "The sentiment of this text is positive.";
//        List<String> sentences = Arrays.asList(
//                scenario,
//                question,
//                answer,
//                scenario2,
//                question2,
//                answer2
//        );
//        createCas("en", sentences);
//        ArrayList<String> ids_list = new ArrayList<>();
//        ids_list.add("Scenario1");
//        ids_list.add("Question1");
//        ids_list.add("Answer1");
//        ids_list.add("Scenario2");
//        ids_list.add("Question2");
//        ids_list.add("Answer2");
//        int i = 0;
//        Collection<Sentence> sentences_list = JCasUtil.select(cas, Sentence.class);
//        for (Sentence sentence : sentences_list) {
//            Div div = new Div(cas, sentence.getBegin(), sentence.getEnd());
//            div.setId(ids_list.get(i));
//            // if i is even, set the role to "question", otherwise set it to "answer"
//            if (i == 0 || i == 3) {
//                div.setDivType("scenario");
//            } else if (i == 1 || i == 4) {
//                div.setDivType("question");
//            } else {
//                div.setDivType("answer");
//            }
//            div.addToIndexes();
//            i++;
//        }

        composer.run(jCas);
        Collection<EssayScore> all_essay_scores = JCasUtil.select(jCas, EssayScore.class);
        for (EssayScore essayScore : all_essay_scores) {
            System.out.println("Begin:" + essayScore.getBegin());
            System.out.println("End:" + essayScore.getEnd());
            System.out.println("EssayScore:" + essayScore.getValue());
            System.out.println("Name: " + essayScore.getName());
            System.out.println("Reason: " + essayScore.getReason());
            System.out.println("Input Question: " + essayScore.getInputQuestion().getValue());
            System.out.println("Input Answer: " + essayScore.getInputAnswer().getValue());
            System.out.println("Input Scene " + essayScore.getInputScene().getValue());
        }
        Collection<EssayScoreLLM> all_essay_scores_llm = JCasUtil.select(jCas, EssayScoreLLM.class);
        for (EssayScoreLLM essayScoreLLM : all_essay_scores_llm) {
            System.out.println("Begin:" + essayScoreLLM.getBegin());
            System.out.println("End:" + essayScoreLLM.getEnd());
            System.out.println(essayScoreLLM.getModelName() + ": ModelName");
            System.out.println(essayScoreLLM.getContents() + ": Contents");
            System.out.println(essayScoreLLM.getResponse() + ": Response");
            System.out.println(essayScoreLLM.getAdditionalInformation() + ": AdditionalInformation");
            System.out.println(essayScoreLLM.getModel().getModelName() + ": ModelName");
        }

    }
}

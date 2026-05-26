package org.hucompute.textimager.uima.l2sca;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
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

import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.ReadabilityAdvance;
import org.texttechnologylab.annotation.L2SCA;

import static org.junit.Assert.assertEquals;

public class MultiTestL2SCA {
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
                        .withParameter("tool", "Textstat")
        );
        List<String> sentences = Arrays.asList(
                "I hate hate it. How can you do that bad thing to me! HOW!",
                "I very happy to be here. I love this place."
        );

        createCas("en", sentences);

        composer.run(cas);

//        Collection<Readability> all_readability = JCasUtil.select(cas, Readability.class);
//        for (Readability readability : all_readability) {
//            FSArray<AnnotationComment> readability_sentences = readability.getTextReadabilities();
//            for (AnnotationComment comment_i : readability_sentences) {
//                System.out.println(comment_i.getKey() + ":" + comment_i.getValue());
//            }
//        }

        Collection<L2SCA> all_l2sca = JCasUtil.select(cas, L2SCA.class);
        for (L2SCA l2sca : all_l2sca) {
            System.out.println("L2SCA:"+ l2sca.getBegin());
            System.out.println("L2SCA:" + l2sca.getEnd());
            System.out.println("L2SCA:" + l2sca.getValue());
            System.out.println("Code:" + l2sca.getCode());
            System.out.println("TypeName:" + l2sca.getTypeName());
            System.out.println("TypeNumber:" + l2sca.getTypeNumber());
            System.out.println("Measure:" + l2sca.getMeasure());
            System.out.println("Definition:" + l2sca.getDefinition());
            System.out.println("Model:" + l2sca.getModel().getModelName());
        }

//        Collection<ReadabilityAdvance> all_readability_advance = JCasUtil.select(cas, ReadabilityAdvance.class);
//        for (ReadabilityAdvance readabilityAdvance : all_readability_advance) {
//            System.out.println("ReadabilityAdvance:" + readabilityAdvance.getGroupName());
//            FSArray<AnnotationComment> readability_sentences = readabilityAdvance.getTextReadabilities();
//            for (AnnotationComment comment_i : readability_sentences) {
//                System.out.println(comment_i.getKey() + ":" + comment_i.getValue());
//            }
//        }
    }
}

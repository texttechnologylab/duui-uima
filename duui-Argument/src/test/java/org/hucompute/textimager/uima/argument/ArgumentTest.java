package org.hucompute.textimager.uima.argument;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
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

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.UnknownHostException;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ArgumentTest {
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

    public void createCas(String language, List<String> sentences) throws UIMAException {
        cas.setDocumentLanguage(language);

        StringBuilder sb = new StringBuilder();
        for (String sentence : sentences) {
            Annotation annotation_i = new Annotation(cas, sb.length(), sb.length()+sentence.length());
            annotation_i.addToIndexes();
            sb.append(sentence).append(" ");
        }

        cas.setDocumentText(sb.toString());
    }

    @Test
    public void MultlingualTest() throws Exception {
        List<String> sentences = Arrays.asList(
                "A zoo is a facility in which all animals are housed within enclosures, displayed to the public, and in which they may also breed.",
                "Zoos produce helpful scientific research.",
                "Zoos save species from extinction and other dangers.",
                "Zoo confinement is psychologically damaging to animals."
        );
        createCas("de", sentences);
//        Collection<Language> languagesall = JCasUtil.select(cas, Language.class);
//        for (Language language_i : languagesall) {
//
//            System.out.println(language_i.getValue());
//        }
//        System.out.println("h");

        composer.add(
                new DUUIRemoteDriver.Component(url)
                        .withParameter("model_name", model)
                        .withParameter("topic", "Zoo")
                        .withParameter("chatgpt_key", "")
        );

        composer.run(cas);
        composer.shutdown();


        Collection<Argument> arguments = JCasUtil.select(cas, Argument.class);
        for (Argument argument_i : arguments) {
            System.out.println(argument_i);
        }
        assertEquals(sentences.size()+1, arguments.size());

    }
}

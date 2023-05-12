package org.texttechnologylab.duui.test;

import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.tcas.Annotation;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.luaj.vm2.LuaError;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.InvalidObjectException;
import java.net.UnknownHostException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertThrows;

@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class TestSuccessNer {
    private DUUIComposer composer;

    @BeforeAll
    void initialize() throws Exception {
        this.composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        this.composer.addDriver(new DUUIRemoteDriver(100));
        this.composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withParameter("language", "de")
        );
    }

    @AfterAll
    void shutdown() throws UnknownHostException {
        this.composer.shutdown();
    }

    private static void checkResults(JCas jCas) {
        Collection<NamedEntity> annotations = JCasUtil.select(jCas, NamedEntity.class);
        assert annotations.size() > 0;
        for (Annotation annotation : annotations) {
            StringBuffer stringBuffer = new StringBuffer();
            annotation.prettyPrint(2, 0, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("  text: \"" + annotation.getCoveredText() + "\"\n");
        }
    }

    @Test
    void test_washington() throws Exception {
        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                "George Washington ging nach Washington."
        );
        jCas.setDocumentLanguage("de");

        Sentence sentence = new Sentence(jCas, 0, 39);
        sentence.addToIndexes(jCas);

        composer.run(jCas);

        checkResults(jCas);
    }

    @Test
    void test_wikipedia() throws Exception {
        BufferedReader buf = new BufferedReader(new FileReader("src/test/resources/wikipedia.txt"));
        ArrayList<String> lines = buf.lines().map(String::trim).collect(Collectors.toCollection(ArrayList::new));
        ArrayList<Integer> lengths = lines.stream().map(String::length).collect(Collectors.toCollection(ArrayList::new));

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                String.join("\n", lines)
        );
        jCas.setDocumentLanguage("de");

        int offset = 0;
        for (int len : lengths) {
            Sentence sentence = new Sentence(jCas, offset, offset + len);
            sentence.addToIndexes(jCas);
            offset += len + 1;
        }

        composer.run(jCas);

        checkResults(jCas);
    }
}


class TestFailureNer {
    @Test
    public void test_no_lang_param() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(100));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                "George Washington ging nach Washington."
        );

        Sentence sentence = new Sentence(jCas, 0, 39);
        sentence.addToIndexes(jCas);

        assertThrows(LuaError.class, () -> composer.run(jCas));

        composer.shutdown();
    }


    @Test
    void test_wrong_lang() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                )
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(100));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                "George Washington ging nach Washington."
        );
        jCas.setDocumentLanguage("abc");

        Sentence sentence = new Sentence(jCas, 0, 39);
        sentence.addToIndexes(jCas);

        assertThrows(InvalidObjectException.class, () -> composer.run(jCas));

        composer.shutdown();
    }
}

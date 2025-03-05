package org.hucompute.textimager.uima;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import static org.junit.jupiter.api.Assertions.*;

public class DUUIReadabilityTest {
    @Test
    public void testReadability() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        composer.add(
//                new DUUIRemoteDriver.Component("http://localhost:8000")
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-readability:latest")
                        .withScale(1)
                        .build()
                        .withTimeout(1000L)
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentLanguage("de");
        jCas.setDocumentText(
                "Das ist ein Test. Wie ist dieser Text zu lesen? Ist er gut lesbar? Ist er verständlich? Das ist ein Test. Wie ist dieser Text zu lesen? Ist er gut lesbar? Ist er verständlich? Das ist ein Test. Wie ist dieser Text zu lesen? Ist er gut lesbar? Ist er verständlich? Ist er verständlich? Das ist ein Test. Wie ist dieser Text zu lesen? Ist er gut lesbar? Ist er verständlich? Das ist ein Test. Wie ist dieser Text zu lesen? Ist er gut lesbar? Ist er verständlich?"
        );

        composer.run(jCas);
        composer.shutdown();

        for (CategoryCoveredTagged category : JCasUtil.select(jCas, CategoryCoveredTagged.class)) {
//            System.out.println(category.getBegin() + " -- " + category.getEnd());
//            System.out.println(category.getCoveredText());
            System.out.println(category.getValue() + " " + category.getScore());
            System.out.println(category.getTags());
            System.out.println("---");
        }

        assertEquals(
                9,  // 9 readability metrics
                JCasUtil.select(jCas, CategoryCoveredTagged.class).size()
        );
    }
}

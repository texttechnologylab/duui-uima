package org.texttechnologylab.duui.gnfinder;

import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.resource.ResourceInitializationException;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.biofid.gnfinder.MetaData;
import org.texttechnologylab.annotation.biofid.gnfinder.Taxon;

public class TestGNFinderV2 {
    private static JCas getJCas() throws ResourceInitializationException, CASException {
        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                String.join("\n", "Nach Schluß des Congresses ist eine längere Excursion vorgesehen, auf welcher die Inseln an der Küste von Pembrokshire besucht werden.", "Dieser Ausflug dürfte besonders interessant werden, weil sich hier große Brutkolonien von Puffinus puffinus und verschiedener Alcidae befinden.", "Auch Thalassidroma pelagica dürfte hier angetroffen werden.", "Bei günstigem Wetter ist ferner der Besuch einer Brutkolonie von Sula bassana vorgesehen.", "Homo sapiens sapiens."));
        jCas.setDocumentLanguage("de");
        return jCas;
    }

    @Test
    public void test_default() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIDockerDriver(10000));
        composer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-gnfinder-v2:0.1.2")
                        .build()
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        printResults(jCas);
    }

    @Test
    public void test_with_noBayes() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIDockerDriver(10000));
        composer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-gnfinder-v2:0.1.2")
                        .withParameter("noBayes", "true")
                        .build()
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        printResults(jCas);
    }

    @Test
    public void test_with_allMatches() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIDockerDriver(10000));
        composer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-gnfinder-v2:0.1.2")
                        .withParameter("allMatches", "true")
                        // Catalogue of Life and GBIF
                        .withParameter("sources", "[1, 11]")
                        .build()
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        printResults(jCas);
    }

    @Test
    public void test_with_oddsDetails() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIDockerDriver(10000));
        composer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-gnfinder-v2:0.1.2")
                        .withParameter("oddsDetails", "true")
                        .build()
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        printResults(jCas);
    }

    private static void printResults(JCas jCas) {
        System.out.println(jCas.select(MetaData.class).findFirst().get().toString(2));
        System.out.println();

        for (Taxon tx : jCas.select(Taxon.class)) {
            System.out.print(tx.toString(2));
            System.out.println("\n  > coveredText: \"" + tx.getCoveredText() + "\"\n");
            System.out.println();
        }
    }
}

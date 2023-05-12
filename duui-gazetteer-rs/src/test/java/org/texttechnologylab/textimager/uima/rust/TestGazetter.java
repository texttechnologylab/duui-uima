package org.texttechnologylab.textimager.uima.rust;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.GeoNamesEntity;
import org.texttechnologylab.annotation.NamedEntity;
import org.texttechnologylab.annotation.type.Taxon;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.zip.GZIPInputStream;

public class TestGazetter {
    @Test
    public void test_biofid() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withParameter("result_selection", "LastPreferFull")
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                String.join(
                        "\n",
                        "Nach Schluß des Congresses ist eine längere Excursion vorgesehen, auf welcher die Inseln an der Küste von Pembrokshire besucht werden.",
                        "Dieser Ausflug dürfte besonders interessant werden, weil sich hier große Brutkolonien von Puffinus p. puffinus und verschiedener Alcidae befinden.",
                        "Auch Thalassidroma pelagica dürfte hier angetroffen werden.",
                        "Bei günstigem Wetter ist ferner der Besuch einer Brutkolonie von Sula bassana vorgesehen.",
                        "Homo sapiens sapiens."));
        jCas.setDocumentLanguage("de");

        composer.run(jCas);

        composer.shutdown();

        for (Taxon taxon : JCasUtil.select(jCas, Taxon.class)) {
            StringBuffer stringBuffer = new StringBuffer();
            taxon.prettyPrint(2, 0, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("  text: \"" + taxon.getCoveredText() + "\"\n");
        }
    }

    @Test
    public void test_biofid_large() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714"));

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentLanguage("de");

        String filename = "src/test/resources/3736518_orig.txt.gz";
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(new GZIPInputStream((new FileInputStream(filename)))))) {
            String text = reader.lines().collect(Collectors.joining("\n"));
            jCas.setDocumentText(text);

            long start = System.currentTimeMillis();
            composer.run(jCas);
            long end = System.currentTimeMillis();

            ArrayList<Taxon> taxons = new ArrayList<>(JCasUtil.select(jCas, Taxon.class));
            System.out.printf("Gazetteer made %d taxon annotations in %d ms.\n", taxons.size(), end - start);

            for (Taxon taxon : taxons.subList(0, 10)) {
                StringBuffer stringBuffer = new StringBuffer();
                taxon.prettyPrint(2, 0, stringBuffer, true);
                System.out.print(stringBuffer);
                System.out.println("  text: \"" + taxon.getCoveredText() + "\"\n");
            }
        }
        composer.shutdown();
    }

    @Test
    public void test_gnd_geonames() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714"));

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                "Johann Wolfgang Goethe, ab 1782 von Goethe (* 28. August 1749 in Frankfurt am Main; † 22. März 1832 in Weimar, Großherzogtum Sachsen-Weimar-Eisenach), war ein deutscher Dichter und Naturforscher. Er gilt als einer der bedeutendsten Schöpfer deutschsprachiger Dichtung. ");
        jCas.setDocumentLanguage("de");

        composer.run(jCas);

        composer.shutdown();

        for (NamedEntity ne : JCasUtil.select(jCas, NamedEntity.class)) {
            StringBuffer stringBuffer = new StringBuffer();
            ne.prettyPrint(2, 0, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("  text: \"" + ne.getCoveredText() + "\"\n");
        }

        for (GeoNamesEntity ne : JCasUtil.select(jCas, GeoNamesEntity.class)) {
            StringBuffer stringBuffer = new StringBuffer();
            ne.prettyPrint(2, 0, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("  text: \"" + ne.getCoveredText() + "\"\n");
        }
    }
}

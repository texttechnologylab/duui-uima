package org.texttechnologylab.duui.gnfinder.v2;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.MetaDataStringField;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.apache.uima.jcas.cas.FSArray;
import org.apache.uima.jcas.cas.NonEmptyFSList;
import org.apache.uima.resource.ResourceInitializationException;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.biofid.gnfinder.MetaData;
import org.texttechnologylab.annotation.biofid.gnfinder.Taxon;

public class TestGNFinderV2 {
    private static JCas getJCas() throws ResourceInitializationException, CASException {
        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentText(
                String.join("\n", "Nach Schluß des Congresses ist eine längere Excursion vorgesehen, auf welcher die Inseln an der Küste von Pembrokshire besucht werden.", "Dieser Ausflug dürfte besonders interessant werden, weil sich hier große Brutkolonien von Puffinus p. puffinus und verschiedener Alcidae befinden.", "Auch Thalassidroma pelagica dürfte hier angetroffen werden.", "Bei günstigem Wetter ist ferner der Besuch einer Brutkolonie von Sula bassana vorgesehen.", "Homo sapiens sapiens."));
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

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714"));

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        StringBuffer stringBuffer = new StringBuffer();
        MetaData md = jCas.select(MetaData.class).findFirst().get();
        md.prettyPrint(0, 2, stringBuffer, true);
        System.out.print(stringBuffer);

        for (Taxon tx : jCas.select(Taxon.class)) {
            stringBuffer = new StringBuffer();
            tx.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  text: \"" + tx.getCoveredText() + "\"\n");
        }
    }


    @Test
    public void test_with_noBayes() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary())
                .withSkipVerification(true);

        composer.addDriver(new DUUIRemoteDriver(10000));
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withParameter("noBayes", "true")
        );

        JCas jCas = getJCas();
        composer.run(jCas);
        composer.shutdown();

        new FSArray<>(jCas, 123);

        StringBuffer stringBuffer;

        stringBuffer = new StringBuffer();
        MetaData md = jCas.select(MetaData.class).findFirst().get();
        md.prettyPrint(0, 2, stringBuffer, true);
        System.out.print(stringBuffer);

        for (Taxon tx : jCas.select(Taxon.class)) {
            stringBuffer = new StringBuffer();
            tx.prettyPrint(0, 2, stringBuffer, true);
            System.out.print(stringBuffer);
            System.out.println("\n  text: \"" + tx.getCoveredText() + "\"\n");
        }
    }

}

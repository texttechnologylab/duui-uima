package org.hucompute.textimager.uima.spacy;

import de.tudarmstadt.ukp.dkpro.core.api.anomaly.type.Anomaly;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
//import org.hucompute.textimager.uima.util.XmlFormatter;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.AnomlySpelling;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.util.Collection;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

public class SpaCyMultiTest {
    @Test
    public void multiTestEn() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(
                        new DUUILuaContext()
                                .withJsonLibrary()
                );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver()
                .withTimeout(10000);
        composer.addDriver(dockerDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714")
        );

//        composer.add(new DUUIDockerDriver.Component("textimager_duui_spellcheck:0.1.3"));

        JCas cas = JCasFactory.createText("Ich habe im Landtag3 L34t3 angesprochen , ich Int3ll3g3nt halte!");
        cas.setDocumentLanguage("de");

        Token t1 = new Token(cas, 0, 3);
        t1.addToIndexes();
        Token t2 = new Token(cas, 4, 8);
        t2.addToIndexes();
        Token t3 = new Token(cas, 9, 11);
        t3.addToIndexes();
        Token t4 = new Token(cas, 12, 20);
        t4.addToIndexes();
        Token t5 = new Token(cas, 21, 26);
        t5.addToIndexes();
        Token t6 = new Token(cas, 27, 39);
        t6.addToIndexes();
        Token t7 = new Token(cas, 40, 41);
        t7.addToIndexes();
        Token t8 = new Token(cas, 42, 45);
        t8.addToIndexes();
        Token t9 = new Token(cas, 46, 57);
        t9.addToIndexes();
        Token t10 = new Token(cas, 58, 63);
        t10.addToIndexes();
        Token t11 = new Token(cas, 64, 64);
        t11.addToIndexes();
        Sentence sent1 = new Sentence(cas, 0, 64);
        sent1.addToIndexes();

        composer.run(cas);

        composer.shutdown();

//        System.out.println(XmlFormatter.getPrettyString(cas));
        Collection<AnomlySpelling> allSpellings = JCasUtil.select(cas, AnomlySpelling.class);
        System.out.println("H");
    }
}

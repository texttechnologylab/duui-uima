package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Paragraph;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token;
import org.apache.uima.cas.CAS;
import org.apache.uima.cas.SerialFormat;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasCopier;
import org.apache.uima.util.CasIOUtils;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.TAscore;
import org.texttechnologylab.uima.type.spacy.SpacyToken;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public class DUUICohMetrixTest {
    @Test
    public void testCohMetrix() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        // TA
//        composer.add(
//                new DUUIRemoteDriver.Component("http://geltlin.hucompute.org:9736")
////                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-coh-metrix:latest")
//                        .withScale(1)
//                        .withSourceView("text")
//                        .build()
//                        .withTimeout(10000L)
//        );

        // Coh-Metrix
        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:8000")
//                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-coh-metrix:latest")
                        .withScale(1)
                        .withSourceView("spacyExt")
                        .build()
                        .withTimeout(100L)
        );

        JCas jCas = JCasFactory.createJCas();
        CasIOUtils.load(
//                new GZIPInputStream(new FileInputStream("/storage/projects/CORE/azure/core-edutec-fileshare/downloads_xmi_ocr_spacy_readability/4b9dc966-cd8f-4f6d-990e-6afbd9133be9/17522/7789.pdf.gz.xmi.gz")),
//                new GZIPInputStream(new FileInputStream("/storage/projects/CORE/projects2/Aclosed/single_xmis/uce_export_2025_06_30/raw/0a8fd26e-5663-49cf-93b8-fb9c01abe8f5/17363/2530020.html.gz.xmi.gz")),
                new GZIPInputStream(new FileInputStream("/storage/projects/CORE/projects2/Aclosed/single_xmis/uce_export_2025_06_30/raw_spacyExt/0e08eef9-b481-44d3-a223-3ce8a24416e4/20593/3083268.html.gz.xmi.gz.xmi.gz")),
                jCas.getCas()
        );

        // TODO copy all needed to "spacyExt" view
        JCas spacyExtView = jCas.getView("spacyExt");
        CasCopier copier = new CasCopier(jCas.getCas(), spacyExtView.getCas());

        spacyExtView.setDocumentLanguage(jCas.getDocumentLanguage());
        spacyExtView.setDocumentText(jCas.getDocumentText());

        for (Paragraph paragraph : JCasUtil.select(jCas, Paragraph.class)) {
            Paragraph copied = copier.copyFs(paragraph);
            copied.addToIndexes();
        }

        System.out.println(jCas.getViewName());
        System.out.println(spacyExtView.getViewName());
        System.out.println(jCas.getView("spacyExt").getView("_InitialView").getViewName());

        for (Paragraph paragraph : JCasUtil.select(spacyExtView, Paragraph.class)) {
            System.out.println("p " + paragraph.getCoveredText());
            for (Sentence sentence : JCasUtil.selectCovered(Sentence.class, paragraph)) {
                System.out.println("s   " + sentence.getCoveredText());
                for (Token token : JCasUtil.selectCovered(SpacyToken.class, sentence)) {
                    System.out.println("t     " + token.getCoveredText());
                }
            }
        }

//        TAscore taScore = new TAscore(jCas);
//        System.out.println("TAscore: " + taScore.getType().getName());

        composer.run(jCas);

//        Path outPath = Paths.get("test_4b9dc966-cd8f-4f6d-990e-6afbd9133be9_17522_7789.pdf.gz.xmi.gz");
//        Path outPath = Paths.get("test_0a8fd26e-5663-49cf-93b8-fb9c01abe8f5_17363_2530020.html.gz.xmi.gz");
        Path outPath = Paths.get("test_0e08eef9-b481-44d3-a223-3ce8a24416e4_20593_3083268.html.gz.xmi.gz");
        try(GZIPOutputStream outputStream = new GZIPOutputStream(Files.newOutputStream(outPath))) {
            CasIOUtils.save(jCas.getCas(), outputStream, SerialFormat.XMI_1_1_PRETTY);
        }
    }
}

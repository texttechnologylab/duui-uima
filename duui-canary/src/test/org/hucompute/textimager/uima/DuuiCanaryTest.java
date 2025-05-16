package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Base64;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class DuuiCanaryTest {
    @Test
    public void testCanary() throws Exception {
        String lang = "de";
        String mimeType = "audio/m4a";
        Path audioPath = Paths.get("/storage/projects/chatgpt-creativity/data/experiments2/Z210_2024-12-16_15-00/test_output3.m4a");

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentLanguage(lang);

        byte[] audioBytes = Files.readAllBytes(audioPath);
        String audioBase64 = Base64.getEncoder().encodeToString(audioBytes);
        jCas.setSofaDataString(audioBase64, mimeType);

        DocumentMetaData meta = DocumentMetaData.create(jCas);
        meta.setLanguage(lang);
        meta.setDocumentBaseUri(audioPath.getParent().toString());
        meta.setCollectionId(audioPath.getParent().toString());
        meta.setDocumentUri(audioPath.toString());
        meta.setDocumentId(audioPath.getFileName().toString());
        meta.setDocumentTitle(audioPath.getFileName().toString());

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);
        DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
        composer.addDriver(uimaDriver);
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withScale(1)
                        .withTargetView("transcript")
//                        .withParameter("language", lang)
                        .withParameter("model", "nvidia/canary-1b-flash")
                        .build()
                        .withTimeout(100000000000L)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class
                , XmiWriter.PARAM_TARGET_LOCATION, "./temp_out"
                , XmiWriter.PARAM_PRETTY_PRINT, true
                , XmiWriter.PARAM_OVERWRITE, true
                , XmiWriter.PARAM_VERSION, "1.1"
                , XmiWriter.PARAM_COMPRESSION, "GZIP"
        )).build());

        composer.run(jCas, "testCanary");
        composer.shutdown();
    }
}

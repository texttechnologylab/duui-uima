package org.hucompute.textimager.uima;

import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.pdf.DUUIPDFReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class TestPDF {
    @Test
    public void testPdf() throws Exception {
        CollectionReaderDescription reader = CollectionReaderFactory.createReaderDescription(
                DUUIPDFReader.class
                , DUUIPDFReader.PARAM_LANGUAGE, "en"
//				, DUUIPDFReader.PARAM_SOURCE_LOCATION, "/home/daniel/Downloads/2410.14626v1.pdf"
                , DUUIPDFReader.PARAM_SOURCE_LOCATION, "/home/daniel/Downloads/PublicWaterMassMailing.pdf"
        );

        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
        composer.addDriver(uimaDriver);
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        composer.add(
//				new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-pdf-extract-ocr:latest")
//				new DUUIRemoteDriver.Component("http://localhost:8000")
                new DUUIRemoteDriver.Component("http://localhost:9714")
//						.withParameter("ocr_preprocess", String.valueOf(true))
                        .withParameter("ocr_dpi", String.valueOf(300))
                        .withParameter("min_chars", String.valueOf(500))
                        .withTargetView("text")
                        .withScale(1)
                        .build()
                        .withTimeout(1000L)
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "./temp_out",
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1",
                XmiWriter.PARAM_COMPRESSION, "GZIP"
        )).build());

        composer.run(reader);
    }
}
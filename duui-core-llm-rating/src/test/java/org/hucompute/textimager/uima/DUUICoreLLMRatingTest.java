package org.hucompute.textimager.uima;

import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.json.JSONObject;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.type.llm.prompt.Message;
import org.texttechnologylab.type.llm.prompt.Prompt;
import org.xml.sax.SAXException;


import java.io.*;
import java.net.URISyntaxException;
import java.util.Iterator;
import java.util.zip.GZIPInputStream;

public class DUUICoreLLMRatingTest {
    @Test
    public void testPrompt() throws Exception {
        long RANDOM_SEED = 1732975931;

        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);

        JSONObject llmArgsJson = new JSONObject();
        llmArgsJson.put("base_url", "localhost:12438");
        llmArgsJson.put("model", "llama3.2:3b-instruct-q4_K_M");
        llmArgsJson.put("temperature", 1);
        llmArgsJson.put("num_ctx", 2048);
        llmArgsJson.put("num_predict", -2);
        llmArgsJson.put("seed", RANDOM_SEED);
        llmArgsJson.put("runs", 5);
        llmArgsJson.put("keep_alive", 3600);

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:8000")
                        .withParameter("llm_args", llmArgsJson.toString())
                        .build()
                        .withTimeout(1000L)
        );

        JCas jCas = JCasFactory.createJCas();
        CasIOUtils.load(
                new GZIPInputStream(new FileInputStream("/storage/projects/baumartz/jcal_2024_textannotator/xmi/eco_nudging/00_export/31491.xmi.gz")),
                jCas.getCas()
        );

        for (Prompt prompt : JCasUtil.select(jCas, Prompt.class)) {
            System.out.println(prompt.getVersion());
            Iterator<Message> it = prompt.getMessages().iterator();
            while (it.hasNext()) {
                Message message = it.next();
                System.out.println(message.getRole());
            }
        }

        composer.run(jCas);
        composer.shutdown();
    }

//    @AfterEach
//    public void afterEach() throws IOException, SAXException {
//        composer.resetPipeline();
//
//        ByteArrayOutputStream stream = new ByteArrayOutputStream();
//        XmlCasSerializer.serialize(cas.getCas(), null, stream);
//        System.out.println(stream.toString(StandardCharsets.UTF_8));
//
//        cas.reset();
//    }
}

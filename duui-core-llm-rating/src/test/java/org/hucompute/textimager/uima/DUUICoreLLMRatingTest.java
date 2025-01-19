package org.hucompute.textimager.uima;

import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.json.JSONObject;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.type.llm.prompt.Prompt;
import org.texttechnologylab.type.llm.prompt.Result;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.io.*;
import java.util.zip.GZIPInputStream;

public class DUUICoreLLMRatingTest {
    @Test
    public void testPrompt() throws Exception {
        long RANDOM_SEED = 1732975931;
        int runs = 2;

        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        //DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        //composer.addDriver(remoteDriver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        JSONObject llmArgsJson = new JSONObject();
//        llmArgsJson.put("base_url", "localhost:11434");
        llmArgsJson.put("base_url", "172.17.0.1:11434");
        llmArgsJson.put("model", "llama3.2:3b-instruct-q4_K_M");
        llmArgsJson.put("temperature", 1);
        llmArgsJson.put("num_ctx", 2048);
        llmArgsJson.put("num_predict", -2);
        llmArgsJson.put("seed", RANDOM_SEED);
        llmArgsJson.put("runs", runs);
        //llmArgsJson.put("keep_alive", 3600);

        composer.add(
                //new DUUIRemoteDriver.Component("http://localhost:8000")
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-core-llm-rating:latest")
                        .withParameter("llm_args", llmArgsJson.toString())
                        .withScale(1)
                        .build()
                        .withTimeout(1000L)
        );

        JCas jCas = JCasFactory.createJCas();
        CasIOUtils.load(
                new GZIPInputStream(new FileInputStream("/storage/projects/baumartz/jcal_2024_textannotator/xmi/eco_nudging/00_export/31491.xmi.gz")),
                jCas.getCas()
        );

        composer.run(jCas);
        composer.shutdown();

        // 1 output result each run for each input prompt
        assertEquals(
                JCasUtil.select(jCas, Prompt.class).size()*runs,
                JCasUtil.select(jCas, Result.class).size()
        );

        // result references prompt
        Prompt prompt = JCasUtil.select(jCas, Prompt.class).iterator().next();
        for (Result result : JCasUtil.select(jCas, Result.class)) {
            assertEquals(result.getPrompt(), prompt);
        }
    }
}

package org.hucompute.textimager.uima;

import org.apache.uima.collection.CollectionReaderDescription;
import org.apache.uima.fit.factory.CollectionReaderFactory;
import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.dkpro.core.io.xmi.XmiReader;
import org.dkpro.core.io.xmi.XmiWriter;


import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class Jcal24Test {
    @Test
    public void testPrompt() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);
        DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
        composer.addDriver(uimaDriver);

        long RANDOM_SEED = 1732975931;
        int runs = 5;
//        String model = "deepseek-r1:70b";
        String model = "llama3.2:3b-instruct-q4_K_M";

        JSONObject llmArgsJson = new JSONObject();
        llmArgsJson.put("base_url", "gondor.hucompute.org:12440");
        llmArgsJson.put("model", model);
        llmArgsJson.put("temperature", 0.8);
        llmArgsJson.put("num_ctx", 2048);
        llmArgsJson.put("num_predict", -2);
        llmArgsJson.put("seed", RANDOM_SEED);
        llmArgsJson.put("runs", runs);
        llmArgsJson.put("keep_alive", 3600);

        composer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-core-llm-rating:0.0.2")
                        .withParameter("llm_args", llmArgsJson.toString())
                        .withScale(1)
                        .build()
                        .withTimeout(1000000000L)
        );

        Path baseDir = Paths.get("/storage/projects/baumartz/jcal_2024_textannotator/xmi/");
        String promptVersion = "v10";
        String task = "eco_nudging";
        String reasoningContexts = "PB";

        Path inDir = baseDir
                .resolve(task)
                .resolve("00_export_prompt_" + promptVersion);

        Path outDir = baseDir
                .resolve(task)
                .resolve("00_llm_prompt_" + promptVersion)
                .resolve(task + "_" + reasoningContexts)
                .resolve("prompt_" + promptVersion)
                .resolve(model.replaceAll("/", "_").replace(":", "_"))
                .resolve("runs" + runs + "_v2");

//        CollectionReaderDescription reader = CollectionReaderFactory.createReaderDescription(XmiReader.class
//                , XmiReader.PARAM_SOURCE_LOCATION, inDir.toString()
//                , XmiReader.PARAM_PATTERNS, "[+]**/*.xmi.gz"
//                , XmiReader.PARAM_ADD_DOCUMENT_METADATA, false
//        );

        DUUIAsynchronousProcessor reader = new DUUIAsynchronousProcessor(new DUUIFileReader(
                inDir.toString(),
                ".xmi.gz",
                1,
                0,
                false,
                "",
                false,
                "de",
                0,
                outDir.toString(),
                ".xmi.xmi.gz"
        ));

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class
                , XmiWriter.PARAM_TARGET_LOCATION, outDir.toString()
                , XmiWriter.PARAM_PRETTY_PRINT, true
                , XmiWriter.PARAM_OVERWRITE, false
                , XmiWriter.PARAM_VERSION, "1.1"
                , XmiWriter.PARAM_COMPRESSION, "GZIP"
        )).build());

        composer.run(reader, "llm");
        composer.shutdown();
    }
}

package org.hucompute.textimager.uima;

import org.json.JSONObject;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.dkpro.core.io.xmi.XmiWriter;


import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class Jcal24Test {
    @Test
    public void testPrompt() throws Exception {
        long RANDOM_SEED = 1732975931;

        int runs = 5;
        for (int run = 0; run < runs; run++) {
            long local_seed = RANDOM_SEED + run;

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

            String model = "llama3.2:3b-instruct-q4_K_M";
            // String model = "gemma2:27b-instruct-q4_0";
            // String model = "llama3.3:70b-instruct-q4_K_M";
            // String model = "nemotron:70b-instruct-q4_K_M";
            // String model = "deepseek-r1:70b";
            // String model = "mistral:7b-instruct-v0.3-q4_0";
            // String model = "mixtral:8x7b-instruct-v0.1-q4_0";
            // String model = "llama3.2:3b-instruct-fp16";
            // String model = "minicpm-v:8b-2.6-fp16";

            JSONObject llmArgsJson = new JSONObject();
            llmArgsJson.put("base_url", "gondor.hucompute.org:12440");
            llmArgsJson.put("model", model);
            llmArgsJson.put("temperature", 0.8);
            llmArgsJson.put("num_ctx", 2048);
            llmArgsJson.put("num_predict", -2);
            llmArgsJson.put("seed", local_seed);
            llmArgsJson.put("keep_alive", 3600);
            llmArgsJson.put("format", "json");

            composer.add(
//                    new DUUIRemoteDriver.Component("http://localhost:8000")
                            new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-core-llm-rating:0.0.3")
                            .withParameter("llm_args", llmArgsJson.toString())
                            .withScale(1)
                            .build()
                            .withTimeout(1000000000L)
            );

            Path baseDir = Paths.get("/storage/projects/baumartz/jcal_2024_textannotator/xmi/");
            String promptName = "ECO_NUDGING_SIMPLE";
            String promptVersion = "10";
            String task = "eco_nudging";
            String reasoningContexts = "PB";

            Path inDir = baseDir
                    .resolve(task)
                    .resolve("01_export")
                    .resolve(promptName)
                    .resolve(promptVersion);

            Path outDir = baseDir
                    .resolve(task)
                    .resolve("01_llm")
                    .resolve(task + "_" + reasoningContexts)
                    .resolve(promptName)
                    .resolve(promptVersion)
                    .resolve(model.replaceAll("/", "_").replace(":", "_"))
                    .resolve("run_" + run);

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
}

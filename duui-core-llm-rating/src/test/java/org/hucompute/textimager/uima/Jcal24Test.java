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

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.concurrent.TimeUnit;
import org.dkpro.core.io.xmi.XmiWriter;
import org.texttechnologylab.DockerUnifiedUIMAInterface.pipeline_storage.sqlite.DUUISqliteStorageBackend;


import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class Jcal24Test {
    @Test
    public void testPrompt() throws Exception {
        long RANDOM_SEED = 1732975931;

        long now = Instant.now().getEpochSecond();

        int runs = 5;
        for (int run = 0; run < runs; run++) {
            long local_seed = RANDOM_SEED + run;

            // String model = "llama3.2:3b-instruct-q4_K_M";
            // String model = "gemma2:27b-instruct-q4_0";
            String model = "llama3.3:70b-instruct-q4_K_M";
            // String model = "nemotron:70b-instruct-q4_K_M";
            //String model = "deepseek-r1:70b";
            // String model = "mistral:7b-instruct-v0.3-q4_0";
            // String model = "mixtral:8x7b-instruct-v0.1-q4_0";
            // String model = "llama3.2:3b-instruct-fp16";
            // String model = "minicpm-v:8b-2.6-fp16";

            Path baseDir = Paths.get("/storage/projects/baumartz/jcal_2024_textannotator/xmi/");
            //String promptName = "ECO_NUDGING_CHAIN_TEXT";
//            String promptName = "ECO_NUDGING_SIMPLE_OPT";
            String promptName = "ECO_NUDGING_SIMPLE_SCHULUNG";
            String promptVersion = "10";
            String task = "eco_nudging";
            String reasoningContexts = "PB";

            Path inDir = baseDir
                    .resolve(task)
                    .resolve("01_export")
                    .resolve(promptName)
                    .resolve(promptVersion);

            String optimRater = "random";

            //String schulungPrompt = "default";
            //String schulungPrompt = "verysimple";
            String schulungPrompt = "simple";
            String schulungModel = "llama3.3_70b-instruct-q4_K_M";
            //String schulungModel = "deepseek-r1_70b";
            //String schulungModel = "nemotron_70b-instruct-q4_K_M";

            if (promptName.equals("ECO_NUDGING_SIMPLE_OPT")) {
                inDir = inDir
                        .resolve(model.replaceAll("/", "_").replace(":", "_"))
                        .resolve(optimRater);
            }
            else if (promptName.equals("ECO_NUDGING_SIMPLE_SCHULUNG")) {
                inDir = inDir
                        .resolve(schulungModel)
                        .resolve(schulungPrompt);
            }
            System.out.println(inDir);

            Path outDir = baseDir
                    .resolve(task)
                    .resolve("01_llm")
                    .resolve(task + "_" + reasoningContexts)
                    .resolve(promptName)
                    .resolve(promptVersion)
                    .resolve(model.replaceAll("/", "_").replace(":", "_"));

            if (promptName.equals("ECO_NUDGING_SIMPLE_OPT")) {
                outDir = outDir
                        .resolve(optimRater);
            }
            else if (promptName.equals("ECO_NUDGING_SIMPLE_SCHULUNG")) {
                outDir = outDir
                        .resolve(schulungModel)
                        .resolve(schulungPrompt);
            }

            outDir = outDir
                    .resolve("run_" + run);
            System.out.println(outDir);
            Files.createDirectories(outDir);

            Path sqlitePath = outDir.resolve("_stats.db");
            DUUISqliteStorageBackend sqlite = new DUUISqliteStorageBackend(sqlitePath.toString())
                    .withConnectionPoolSize(1);

            DUUIComposer composer = new DUUIComposer()
                    .withWorkers(1)
                    .withSkipVerification(true)
                    .withStorageBackend(sqlite)
                    .withLuaContext(new DUUILuaContext().withJsonLibrary());

            DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
            composer.addDriver(dockerDriver);
            DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
            composer.addDriver(uimaDriver);
            DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
            composer.addDriver(remoteDriver);

            String baseUrl = "gondor.hucompute.org:12440";

            JSONObject llmArgsJson = new JSONObject();
            llmArgsJson.put("base_url", baseUrl);
            llmArgsJson.put("system", baseUrl);  // in case we use local port forwarding or similar...
            llmArgsJson.put("model", model);
            llmArgsJson.put("temperature", 0.8);
            if (promptName.equals("ECO_NUDGING_SIMPLE_SCHULUNG")) {
                llmArgsJson.put("num_ctx", 128000);
            }
            else {
                llmArgsJson.put("num_ctx", 2048);
            }
            llmArgsJson.put("num_predict", -2);
            llmArgsJson.put("seed", local_seed);
            llmArgsJson.put("keep_alive", 3600);
            llmArgsJson.put("format", "json");

            composer.add(
//                    new DUUIRemoteDriver.Component("http://localhost:8000")
                    new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-core-llm-rating:0.0.4")
                            .withParameter("llm_args", llmArgsJson.toString())
                            .withScale(1)
                            .build()
                            .withTimeout(1000000000L)
            );

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

            composer.run(reader, "llm_"+String.valueOf(now));
            composer.shutdown();

            // give time to write final output
            TimeUnit.MINUTES.sleep(3);
        }
    }
}

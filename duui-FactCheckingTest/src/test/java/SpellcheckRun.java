import org.apache.uima.analysis_engine.AnalysisEngineDescription;
import org.apache.uima.collection.CollectionReaderDescription;
import org.dkpro.core.api.resources.CompressionMethod;
import org.dkpro.core.io.xmi.XmiReader;
import org.dkpro.core.io.xmi.XmiWriter;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;

import java.nio.file.Path;
import java.nio.file.Paths;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import static org.apache.uima.fit.factory.CollectionReaderFactory.createReaderDescription;
public class SpellcheckRun {
    static public void main(String[] args) throws Exception {
        Path inputDir = Paths.get("/storage/projects/bagci/test/gerparcor/xmi_spellchecks3/Germany");
        Path outputDir = Paths.get("/storage/projects/bagci/test/gerparcor/xmi_sentiment");

        CollectionReaderDescription reader = createReaderDescription(XmiReader.class
                , XmiReader.PARAM_SOURCE_LOCATION, inputDir.toString()
                , XmiReader.PARAM_PATTERNS, "**/*.xmi.gz"
        );

        AnalysisEngineDescription writer = createEngineDescription(XmiWriter.class
                , XmiWriter.PARAM_TARGET_LOCATION, outputDir.toString()
                , XmiWriter.PARAM_VERSION, "1.1"
                , XmiWriter.PARAM_COMPRESSION, CompressionMethod.GZIP
                , XmiWriter.PARAM_PRETTY_PRINT, true
                , XmiWriter.PARAM_OVERWRITE, true
        );

        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

//        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
//        composer.addDriver(dockerDriver);
//
//        DUUIUIMADriver uimaDriver = new DUUIUIMADriver();
//        composer.addDriver(uimaDriver);
//
//        composer.add(
//                new DUUIDockerDriver.Component("http://127.0.0.1:9714")
//                        .withImageFetching()
//                        .withScale(1)
//                        .withRunningAfterDestroy(true)
//        );

        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver(10000);
        composer.addDriver(remote_driver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver()
                .withTimeout(10000);
        composer.addDriver(dockerDriver);
        composer.addDriver(new DUUIUIMADriver());
        composer.add(
                new DUUIRemoteDriver.Component("http://127.0.0.1:9714").
                        withParameter("model_name", "cardiffnlp/twitter-xlm-roberta-base-sentiment").
                        withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence")
        );

        composer.add(new DUUIUIMADriver.Component(writer));
        composer.run(reader);
        composer.shutdown();
    }
}

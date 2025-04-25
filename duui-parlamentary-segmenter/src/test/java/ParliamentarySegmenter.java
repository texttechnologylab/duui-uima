import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.InvalidXMLException;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.DUUIAsynchronousProcessor;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReaderLazy;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.writer.TTLabXmiWriter;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.net.URISyntaxException;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

/**
 * ParliamentrySegmenter - Test
 * @author Giuseppe Abrami
 */
public class ParliamentarySegmenter {

    private static DUUIComposer pComposer = null;

    private static int iWorkers = 1;

    @BeforeAll
    public static void prepare() throws IOException, URISyntaxException, UIMAException, SAXException {

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        pComposer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remote_driver = new DUUIRemoteDriver();

        // Hinzufügen der einzelnen Driver zum Composer
        pComposer.addDriver(uima_driver, docker_driver, remote_driver);

    }

    @Test
    public void test() throws Exception {

        DUUIFileReaderLazy pFileReader = new DUUIFileReaderLazy("/home/gabrami/Downloads/WR", ".xmi.gz", 10);

        DUUIAsynchronousProcessor pProcessor = new DUUIAsynchronousProcessor(pFileReader);

        pComposer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
//                new DUUIDockerDriver.Component("entailab.docker.texttechnologylab.org/duui-parliament-segmenter:0.1")
//                        .withImageFetching()
                        .withScale(iWorkers)
                        .build()
        );

        pComposer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "/tmp/",
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1",
                XmiWriter.PARAM_PRETTY_PRINT, true

        )).withScale(iWorkers)
          .build());

        pComposer.run(pProcessor, "testrun");

    }

    @Test
    public void extractPlain() throws Exception {

        DUUIFileReaderLazy pFileReader = new DUUIFileReaderLazy("/path", ".xmi.gz", 10);

        DUUIAsynchronousProcessor pProcessor = new DUUIAsynchronousProcessor(pFileReader);

        pComposer.add(
                new DUUIDockerDriver.Component("entailab.docker.texttechnologylab.org/duui-parliament-segmenter:0.1")
//                        .withImageFetching()
                        .withScale(iWorkers)
                        .build()
        );

        pComposer.add(new DUUIUIMADriver.Component(createEngineDescription(PlainWriter.class,
                PlainWriter.PARAM_TARGET_LOCATION, "/tmp/",
                PlainWriter.PARAM_FILENAME_EXTENSION, ".txt",
                PlainWriter.PARAM_OVERWRITE, true
        )).withScale(iWorkers)
                .build());

        pComposer.run(pProcessor, "testrun");

    }

}

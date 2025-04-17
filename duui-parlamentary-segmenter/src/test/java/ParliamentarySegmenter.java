import org.apache.commons.compress.compressors.CompressorException;
import org.apache.uima.UIMAException;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.reader.DUUIFileReaderLazy;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.net.URISyntaxException;

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

        // Hinzufügen der einzelnen Driver zum Composer
        pComposer.addDriver(uima_driver, docker_driver);

    }

    @Test
    public void test() throws URISyntaxException, IOException, CompressorException, InvalidXMLException, SAXException {

        DUUIFileReaderLazy pFileReader = new DUUIFileReaderLazy("/targetpath", ".xmi.gz", 10);

        pComposer.add(
                new DUUIDockerDriver.Component("entailab.docker.texttechnologylab.org/duui-parliament-segmenter:latest")
                        .withImageFetching()
                        .build()
        );

        pComposer.add(
                new DUUIDockerDriver.Component("entailab.docker.texttechnologylab.org/duui-parliament-segmenter:latest")
                        .withImageFetching()
                        .build()
        );

    }

}

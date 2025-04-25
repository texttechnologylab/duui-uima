import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class YouTubeDownloader {

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
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        docker_driver.withTimeout(60);


        // Hinzufügen der einzelnen Driver zum Composer
        pComposer.addDriver(uima_driver, docker_driver, remoteDriver);

    }

    @Test
    public void test() throws Exception {

        try {
            ClassLoader classLoader = YouTubeDownloader.class.getClassLoader();
            URL fCookies = classLoader.getResource("cookies.txt");
            String sCookies = Files.readString(Path.of(fCookies.toURI()));
        }
        catch (Exception e){
            System.err.println("No Cookies, but no problem at all.");
        }

        JCas pCas = JCasFactory.createJCas();
        pCas.setDocumentText("https://www.youtube.com/watch?v=f3kB9pNPjsk");
        pCas.setDocumentLanguage("en");

        pComposer.add(
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-youtube-downloader:0.1")
                        .withParameter("withTranscription", "true")
//                        .withParameter("cookies", sCookies)   // not nessecary
                        .withTargetView("video")
                        .withScale(iWorkers)
                        .build().withTimeout(60)
        );

        pComposer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "/tmp/",
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1",
                XmiWriter.PARAM_PRETTY_PRINT, true

        )).withScale(iWorkers)
                .build());

        pComposer.run(pCas, "testrun");

    }


}

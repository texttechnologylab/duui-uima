import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.io.FileUtils;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIUIMADriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.writer.TTLabXmiWriter;
import org.dkpro.core.io.xmi.XmiWriter;

import javax.imageio.ImageIO;

public class Video2Audio {

    ClassLoader classLoader = Video2Audio.class.getClassLoader();
    URL fVideo = classLoader.getResource("interview2.mp4");

    private static void saveBase64ToAudio(String base64String, String outputPath) {
        try {
            // Decode the Base64 string into a byte array
            byte[] decodedBytes = Base64.getDecoder().decode(base64String);

            // Save the image to the specified output file
            File outputFile = new File(outputPath);
            Files.write(outputFile.toPath(), decodedBytes);

            System.out.println("Video saved as: " + outputPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Test
    @DisplayName("ExtractionTest")
    public void ExtractionTest() throws Exception {


        DUUIComposer composer = new DUUIComposer().withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIUIMADriver duuiuimaDriver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver, duuiuimaDriver);


        File fFile = new File(fVideo.getPath());
        byte[] bFile = FileUtils.readFileToByteArray(fFile);
        String encodedString = Base64.getEncoder().encodeToString(bFile);
        String pMimeType = Files.probeContentType(Path.of(fVideo.getPath()));

        JCas pCas = JCasFactory.createText("Programm");

        JCas videoView = pCas.createView("video");
        JCas transcript = pCas.createView("transcript");
        videoView.setSofaDataString(encodedString, pMimeType);
        videoView.setDocumentLanguage("de");

        JCas audioView = pCas.createView("audio");

        composer.add(
                new DUUIRemoteDriver.Component("http://localhost:9714")
                        .withTargetView("audio")
                        .withSourceView("video")
                        .withScale(1).build());



        composer.add(new DUUIRemoteDriver.Component("http://whisperx.lehre.texttechnologylab.org")
                .withScale(1)
                .withSourceView("audio")
                .withTargetView("transcript")
                .build()
        );

        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, "/tmp/xmi/",
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_COMPRESSION, "GZIP")).build());

        composer.run(pCas);

        saveBase64ToAudio(audioView.getSofa().getSofaString(), "/tmp/audio/test_audi.mp3");


    }

}

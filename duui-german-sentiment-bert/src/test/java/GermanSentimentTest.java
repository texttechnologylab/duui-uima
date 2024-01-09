import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import io.swagger.models.auth.In;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.util.CasIOUtils;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.luaj.vm2.ast.Str;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.SentimentBert;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class GermanSentimentTest {

    @Test
    public void test() throws Exception {

        int iWorkers = 1;
        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.


        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver(1000);
        composer.addDriver(docker_driver, remoteDriver);


        boolean useDockerImage = true;
        if (useDockerImage){
             composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                    .withScale(iWorkers)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(iWorkers)
                .build());
        }

        // Create basic test jCas.
        JCas jCas = JCasFactory.createText("Dies ist ein super schöner Test. Leider weiß ich nicht weiter.", "de");

        new Sentence(jCas, 0, jCas.getDocumentText().indexOf(".")+1).addToIndexes();
        new Sentence(jCas, jCas.getDocumentText().indexOf(".")+2, jCas.getDocumentText().lastIndexOf(".")).addToIndexes();
         System.out.println(JCasUtil.select(jCas, Sentence.class).size());

        composer.run(jCas, "test");

        // Print Result
        for(SentimentBert sentimentBert: JCasUtil.select(jCas, SentimentBert.class)){
            System.out.println(sentimentBert.getCoveredText() + ": " + sentimentBert.getSentiment());
        }

    }

    @Test
    public void analyseParlamentDataSentimentBert() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/SentimentBert";
        String sSuffix = "xmi.gz";

        int workers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withWorkers(workers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        boolean useDockerImage = false;

        if (useDockerImage){
             composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/german-sentiment-bert:latest")
                    .withScale(workers)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(workers)
                .build());
        }

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, "test");

        // Print the past time in seconds
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    public static String formatNanoSeconds(long nanoSeconds){
        double runTime = Math.round(((double) nanoSeconds) / 10_000_000.0) / 100.0;
        return runTime + "s";
    }


}

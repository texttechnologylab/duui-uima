import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.dkpro.core.io.xmi.XmiWriter;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.io.AsyncCollectionReader;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.hucompute.textimager.uima.type.category.CategoryCoveredTagged;
import org.texttechnologylab.DockerUnifiedUIMAInterface.pipeline_storage.sqlite.DUUISqliteStorageBackend;

import java.io.File;

import static org.apache.uima.fit.factory.AnalysisEngineFactory.createEngineDescription;

public class ZeroShotTest {

    public static final String dijkstraExampleText = "Der Algorithmus von Dijkstra (nach seinem Erfinder Edsger W. Dijkstra) ist ein Algorithmus aus der Klasse der Greedy-Algorithmen[1] und löst das Problem der kürzesten Pfade für einen gegebenen Startknoten. " +
            "Er berechnet somit einen kürzesten Pfad zwischen dem gegebenen Startknoten und einem der (oder allen) übrigen Knoten in einem kantengewichteten Graphen (sofern dieser keine Negativkanten enthält).\n \n" +
            "Für unzusammenhängende ungerichtete Graphen ist der Abstand zu denjenigen Knoten unendlich, zu denen kein Pfad vom Startknoten aus existiert. Dasselbe gilt auch für gerichtete nicht stark zusammenhängende Graphen. Dabei wird der Abstand synonym auch als Entfernung, Kosten oder Gewicht bezeichnet.";

    public static final String labels = "Tiere,Pflanzen,Technologie,Geschichte,Kunst,Musik,Politik,Bildung,Sport,Gesundheit,Reisen,Essen und Trinken,Filme,Literatur,Umwelt,Wissenschaft,Mode,Philosophie,Psychologie,Wirtschaft";

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
             composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/debert-zero-shot-category:latest")
                    .withScale(iWorkers)
                     .withParameter("labels", labels)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                .withScale(iWorkers)
                .withParameter("labels", labels)
                .build());
        }

        // Create basic test jCas.
        JCas jCas = JCasFactory.createText(dijkstraExampleText, "de");

        composer.run(jCas, "test");

        // Print Result
        for(CategoryCoveredTagged categoryCoveredTagged: JCasUtil.select(jCas, CategoryCoveredTagged.class)){
            System.out.println(categoryCoveredTagged.getValue() + ": " + categoryCoveredTagged.getScore());
        }

        //org.texttechnologylab.annotation.twitter.
        //org.hucompute.textimager.uima.type.Similarity
    }

    @Test
    public void analyseParlamentData() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/ThirdReich";
        String sOutputPath = "/home/max/uni/testdata/output/ThirdReich/DeBERTZeroShot";
        String sSuffix = "xmi.gz";

        String runName = "python-image";
        int iWorkers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUISqliteStorageBackend sqlite = new DUUISqliteStorageBackend("pipeline_data.db")
                .withConnectionPoolSize(iWorkers);

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withStorageBackend(sqlite)
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        boolean useDockerImage = false;
        if (useDockerImage){
            composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/debert-zero-shot-category:latest")
                    .withScale(iWorkers)
                    .withParameter("labels", labels)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                    .withScale(iWorkers)
                    .withParameter("labels", labels)
                    .build());
        }

        // Hinzufügen einer UIMA-Componente zum schreiben der Ergebnisse
        composer.add(new DUUIUIMADriver.Component(createEngineDescription(XmiWriter.class,
                XmiWriter.PARAM_TARGET_LOCATION, sOutputPath,
                XmiWriter.PARAM_PRETTY_PRINT, true,
                XmiWriter.PARAM_OVERWRITE, true,
                XmiWriter.PARAM_VERSION, "1.1"
        )).build());

        sqlite.addNewRun(runName, composer);

        long startTime = System.nanoTime();
        composer.run(pCorpusReader, runName);

        // Print the past time in seconds
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    @Test
    public void analyseBadenWuertemberg() throws Exception {

        // Input- und Output-Pfade
        String sInputPath = "/home/max/uni/testdata/input/BadenWuertemberg";
        String sOutputPath = "/home/max/uni/testdata/output/BadenWuertemberg/DeBERTZeroShotCuda";
        String sSuffix = "xmi.gz";

        String runName = "cuda-image";
        int iWorkers = 1;

        // Asynchroner reader für die Input-Dateien
        AsyncCollectionReader pCorpusReader = new AsyncCollectionReader(sInputPath, sSuffix, 1, false);
        new File(sOutputPath).mkdir();

        DUUISqliteStorageBackend sqlite = new DUUISqliteStorageBackend("pipeline_data.db")
                .withConnectionPoolSize(iWorkers);

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        // Instanziierung des Composers, mit einigen Parametern
        DUUIComposer composer = new DUUIComposer()
                .withSkipVerification(true)     // wir überspringen die Verifikation aller Componenten =)
                .withLuaContext(ctx)            // wir setzen den definierten Kontext
                .withStorageBackend(sqlite)
                .withWorkers(iWorkers);         // wir geben dem Composer eine Anzahl an Threads mit.

        sqlite.addNewRun(runName, composer);

        DUUIDockerDriver docker_driver = new DUUIDockerDriver();
        DUUIUIMADriver uima_driver = new DUUIUIMADriver()
                .withDebug(false);
        DUUIRemoteDriver duuiRemoteDriver = new DUUIRemoteDriver();
        composer.addDriver(docker_driver, uima_driver, duuiRemoteDriver);

        boolean useDockerImage = false;
        if (useDockerImage){
            composer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/debert-zero-shot-category:latest")
                    .withScale(iWorkers)
                    .withParameter("labels", labels)
                    .build());
        }else{
            composer.add(new DUUIRemoteDriver.Component("http://localhost:9714")
                    .withScale(iWorkers)
                    .withParameter("labels", labels)
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
        composer.run(pCorpusReader, runName);

        // Print the past time in seconds
        long endTime = System.nanoTime();
        System.out.println("Time passed: " + formatNanoSeconds(endTime-startTime));

    }

    public static String formatNanoSeconds(long nanoSeconds){
        double runTime = Math.round(((double) nanoSeconds) / 10_000_000.0) / 100.0;
        return runTime + "s";
    }


}

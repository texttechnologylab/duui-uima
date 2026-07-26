import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.commons.compress.compressors.CompressorException;
import org.apache.commons.io.IOUtils;
import org.apache.uima.UIMAException;
import org.apache.uima.cas.CASException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.apache.uima.util.InvalidXMLException;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.Embedding;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;

public class EmbeddingTest {

    public static DUUIComposer pComposer = null;
    public static final int WORKERS = 5;

    @BeforeAll
    public static void init() throws IOException, URISyntaxException, UIMAException, SAXException {

        System.out.println("INIT...");

        DUUILuaContext ctx = new DUUILuaContext().withJsonLibrary();

        pComposer = new DUUIComposer()
                .withSkipVerification(true)
                .withLuaContext(ctx)
                .withWorkers(WORKERS);

        DUUIUIMADriver uima_driver = new DUUIUIMADriver();
        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        DUUIPodmanDriver podmanDriver = new DUUIPodmanDriver();
        DUUIKubernetesDriver kubernetesDriver = new DUUIKubernetesDriver();
        DUUISwarmDriver swarmDriver = new DUUISwarmDriver();

        // Hinzufügen der einzelnen Driver zum Composer
        pComposer.addDriver(
                uima_driver,
                remoteDriver,
                dockerDriver,
                podmanDriver,
                kubernetesDriver,
                swarmDriver
        );

        System.out.println("INIT done");

    }

    public JCas getCAS() throws IOException, ResourceInitializationException, CASException {

        InputStream is = EmbeddingTest.class.getClassLoader().getResourceAsStream("input.txt");
        String sInput = IOUtils.toString(is, StandardCharsets.UTF_8);

        JCas pCas = JCasFactory.createText(sInput, "de");

        return pCas;

    }

    @Test
    public void test() throws Exception {

        JCas pCas = getCAS();

        pComposer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-spacy-de_core_news_sm:0.4.1")
                .withImageFetching()
                .build());

        pComposer.add(new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-ollama-embeddings:test")
                .withParameter("apiUrl", "https://llm.texttechnologylab.org/ollama/api/embed")
                .withParameter("model", "anduin.jina/jina-embeddings-v2-base-de:latest")
                .withParameter("apiKey", "myapikey")
                .withParameter("chunkSize", "300")
                .withParameter("selection", Sentence.class.getName())
                .withImageFetching()
                .build());

        pComposer.run(pCas);

        JCasUtil.select(pCas, Embedding.class).stream().forEach(e->{
            System.out.println(e.getBegin());
            System.out.println(e.getEnd());
            System.out.println(e.getCoveredText());
        });

    }

}

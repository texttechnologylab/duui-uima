package org.hucompute.textimager.uima;

import de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.*;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.uima.type.Embedding;

import java.util.ArrayList;
import java.util.List;

public class DUUISentenceTransformersTest {
    @Test
    public void testST() throws Exception {
        DUUIComposer composer = new DUUIComposer()
                .withWorkers(1)
                .withSkipVerification(true)
                .withLuaContext(new DUUILuaContext().withJsonLibrary());

        DUUIRemoteDriver remoteDriver = new DUUIRemoteDriver();
        composer.addDriver(remoteDriver);
        DUUIDockerDriver dockerDriver = new DUUIDockerDriver();
        composer.addDriver(dockerDriver);

        composer.add(
//                new DUUIRemoteDriver.Component("http://localhost:8080")
                new DUUIDockerDriver.Component("docker.texttechnologylab.org/duui-sentence-transformers:latest")
                        .withParameter("model_name", "jinaai/jina-embeddings-v2-base-de")
                        .withParameter("batch_size", String.valueOf(16))
                        .withScale(1)
                        .build()
                        .withTimeout(1000L)
        );

        JCas jCas = JCasFactory.createJCas();
        jCas.setDocumentLanguage("de");

        List<String> sentences = new ArrayList<>();
        sentences.add("Das ist ein Test.");
        sentences.add("Das ist noch ein Test.");

        StringBuilder sofaString = new StringBuilder();
        for (String sentence : sentences) {
            int begin = sofaString.length();
            sofaString.append(sentence);
            int end = sofaString.length();
            sofaString.append(" ");
            Sentence sentenceAnno = new Sentence(jCas, begin, end);
            sentenceAnno.addToIndexes();
        }
        jCas.setDocumentText(sofaString.toString());

        composer.run(jCas);
        composer.shutdown();

        System.out.println(jCas.getDocumentText());

        List<Embedding> embeddings = new ArrayList<>(JCasUtil.select(jCas, Embedding.class));
        assert embeddings.size() == sentences.size();
        assert embeddings.get(0).getEmbedding().size() == 768;
    }
}

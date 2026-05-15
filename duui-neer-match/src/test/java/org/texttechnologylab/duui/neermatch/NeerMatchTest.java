package org.texttechnologylab.duui.neermatch;

import org.apache.uima.UIMAException;
import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPodmanDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.NeerMatchPrediction;
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

public class NeerMatchTest {

	DUUIComposer composer;
	JCas cas1;
	JCas cas2;
	JCas resultCas;

	@BeforeEach
	public void setup() throws IOException, URISyntaxException, UIMAException, SAXException {
		try {
			ClassLoader.getSystemClassLoader().loadClass("org.texttechnologylab.annotation.NeerMatchPrediction");
		} catch (ClassNotFoundException e) {
			throw new RuntimeException(e);
		}
		composer = new DUUIComposer()
			.withSkipVerification(true)
			.withLuaContext(new DUUILuaContext().withJsonLibrary());

		composer.addDriver(new DUUIRemoteDriver(), new DUUIPodmanDriver());

		cas1 = JCasFactory.createJCas();
		cas1.setDocumentLanguage("de");
		cas1.setDocumentText("Das ist ein Test.");

		cas2 = JCasFactory.createJCas();
		cas2.setDocumentLanguage("de");
		cas2.setDocumentText("Dies ist der zweite Test!");

		resultCas = JCasFactory.createJCas();
		resultCas.setDocumentLanguage("de");
		resultCas.setDocumentText("");
	}

	@Test
	public void test() throws Exception {

		composer.add(
			new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
				.build());
		composer.add(new DUUIRemoteDriver.Component("http://localhost:12345")
			.withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
			.withParameter("model", "test3")
			.withParameter("pipeline_id", UUID.randomUUID().toString())
			.withParameter("threshold", "0.1")
			.build()
		);
		composer.run(cas1);
		composer.run(cas2);
		composer.run(resultCas);

		saveXmi(cas1, "cas1.xmi");
		saveXmi(cas2, "cas2.xmi");
		saveXmi(resultCas, "resultCas.xmi");
	}

	void saveXmi(JCas cas, String filename) throws IOException, SAXException {
		String xmi;
		try (var baos = new ByteArrayOutputStream()) {
			XmiCasSerializer.serialize(cas.getCas(), baos);
			xmi = baos.toString();
		}
		Files.createDirectories(Path.of("results"));
		Files.writeString(Path.of("results", filename), xmi);
	}
}

package org.texttechnologylab.duui.neermatch;

import org.apache.uima.UIMAException;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPodmanDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.net.URISyntaxException;

public class NeerMatchTest {

	DUUIComposer composer;
	JCas cas;

	@BeforeEach
	public void setup() throws IOException, URISyntaxException, UIMAException, SAXException {
		composer = new DUUIComposer()
			.withSkipVerification(true)
			.withLuaContext(new DUUILuaContext().withJsonLibrary());

		composer.addDriver(new DUUIRemoteDriver(), new DUUIPodmanDriver());

		cas = JCasFactory.createJCas();
		cas.setDocumentLanguage("de");
		cas.setDocumentText("Das ist ein Test.");
	}

	@Test
	public void test() throws Exception {
		composer.add(
			new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
				.build());
		composer.add(new DUUIRemoteDriver.Component("http://localhost:12345")
			.withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
			.withParameter("model", "test3")
			.withParameter("query", "ein")
			.withParameter("threshold", "0.4")
			.build()
		);
		composer.run(cas);

	}
}

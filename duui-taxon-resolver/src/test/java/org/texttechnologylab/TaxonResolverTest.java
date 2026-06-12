package org.texttechnologylab;

import org.apache.uima.cas.CAS;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaSandbox;

import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;

public class TaxonResolverTest {

	DUUIComposer composer;

	@BeforeEach
	public void setup() throws URISyntaxException, IOException {
		composer = new DUUIComposer()
			.withSkipVerification(true)
			.withLuaContext(new DUUILuaContext().withJsonLibrary());
		composer.addDriver(new DUUIRemoteDriver());
	}

	@Test
	public void testTaxonResolver() throws Exception {
		composer.add(new DUUIRemoteDriver.Component("http://localhost:12321")
			.withParameter("annotations_view", "taxon")
		);

		JCas jcas = JCasFactory.createJCas();
		try (InputStream is = ClassLoader.getSystemClassLoader().getResourceAsStream("4958866.xmi")) {
			CAS cas = jcas.getCas();
			XmiCasDeserializer.deserialize(is, cas);
		} catch (Exception e) {
			throw new Exception("Error deserializing CAS: " + e.getMessage());
		}

		composer.run(jcas);
	}

}

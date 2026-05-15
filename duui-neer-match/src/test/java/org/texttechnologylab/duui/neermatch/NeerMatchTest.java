package org.texttechnologylab.duui.neermatch;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
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
import org.xml.sax.SAXException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.UUID;

public class NeerMatchTest {

	static final String NEER_MATCH_URL = "http://localhost:12345";

	DUUIComposer composer;
	JCas cas1;
	JCas cas2;
	String pipelineId;
	String model = "test3";
	double threshold = 0.1;

	@BeforeEach
	public void setup() throws IOException, URISyntaxException, UIMAException, SAXException {
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

		pipelineId = UUID.randomUUID().toString();
	}

	@Test
	public void test() throws Exception {

		composer.add(
			new DUUIPodmanDriver.Component("docker.texttechnologylab.org/textimager-duui-spacy:0.4.0").withImageFetching()
				.build());
		composer.add(new DUUIRemoteDriver.Component(NEER_MATCH_URL)
			.withParameter("selection", "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token")
			.withParameter("pipeline_id", pipelineId)
			.build()
		);
		composer.run(cas1);
		composer.run(cas2);

		JsonObject result = finalizeNeerMatch();
		Gson gson = new GsonBuilder()
			.setPrettyPrinting()
			.create();
		System.out.println(gson.toJson(result));
	}

	JsonObject finalizeNeerMatch() throws IOException, InterruptedException {
		String response;
		try (HttpClient client = HttpClient.newHttpClient()) {
			JsonObject requestBody = new JsonObject();
			requestBody.addProperty("pipeline_id", pipelineId);
			JsonObject requestProperties = new JsonObject();
			requestProperties.addProperty("model", model);
			requestProperties.addProperty("threshold", threshold);
			requestBody.add("properties", requestProperties);

			response = client.send(
				HttpRequest.newBuilder()
					.uri(java.net.URI.create(NEER_MATCH_URL + "/v1/finalize"))
					.header("Content-Type", "application/json")
					.POST(HttpRequest.BodyPublishers.ofString(requestBody.toString()))
					.build(),
				HttpResponse.BodyHandlers.ofString()
			).body();
		}

		return JsonParser.parseString(response).getAsJsonObject();
	}
}

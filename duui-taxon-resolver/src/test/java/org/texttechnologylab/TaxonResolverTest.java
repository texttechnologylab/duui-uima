package org.texttechnologylab;

import org.apache.uima.cas.impl.XmiCasSerializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.texttechnologylab.DockerUnifiedUIMAInterface.DUUIComposer;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIDockerDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIPodmanDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.driver.DUUIRemoteDriver;
import org.texttechnologylab.DockerUnifiedUIMAInterface.lua.DUUILuaContext;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.RecognizedTaxon;
import org.texttechnologylab.annotation.type.Taxon;
import org.texttechnologylab.annotation.type.TaxonResolution;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;

import static org.junit.jupiter.api.Assertions.*;

public class TaxonResolverTest {

	enum Strategy {
		REMOTE,
		DOCKER,
		PODMAN
	}

	static final String TAXON_RESOLVER_IMAGE = "localhost/duui-taxon-resolver:1.0.0";
	static final String TAXON_RESOLVER_REMOTE_HOST = "http://localhost:12345";
	static final Strategy STRATEGY = Strategy.REMOTE;
	static final boolean STORE_OUTPUT = false;

	DUUIComposer composer;

	@BeforeEach
	public void setup() throws Exception {
		composer = new DUUIComposer()
			.withSkipVerification(true)
			.withLuaContext(new DUUILuaContext().withJsonLibrary());
		switch (STRATEGY) {
			case REMOTE -> composer.addDriver(new DUUIRemoteDriver());
			case DOCKER -> composer.addDriver(new DUUIDockerDriver());
			case PODMAN -> composer.addDriver(new DUUIPodmanDriver());
		}
	}

	private void addTaxonResolver(String annotationsView) throws Exception {
		switch (STRATEGY) {
			case REMOTE -> {
				var component = new DUUIRemoteDriver.Component(TAXON_RESOLVER_REMOTE_HOST);
				if (annotationsView != null) {
					component.withParameter("annotations_view", annotationsView);
				}
				composer.add(component.build());
			}
			case DOCKER -> {
				var component = new DUUIDockerDriver.Component(TAXON_RESOLVER_IMAGE);
				if (annotationsView != null) {
					component.withParameter("annotations_view", annotationsView);
				}
				composer.add(component.build());
			}
			case PODMAN -> {
				var component = new DUUIPodmanDriver.Component(TAXON_RESOLVER_IMAGE);
				if (annotationsView != null) {
					component.withParameter("annotations_view", annotationsView);
				}
				composer.add(component.build());
			}
		}
	}

	@Test
	public void testEmpty() throws Exception {
		addTaxonResolver(null);

		JCas jcas = JCasFactory.createJCas();
		jcas.setDocumentLanguage("en");
		jcas.setDocumentText("This is a test.");

		composer.run(jcas);

		storeCas(jcas, "empty");

		Collection<RecognizedTaxon> taxa = JCasUtil.select(jcas, RecognizedTaxon.class);
		assertTrue(taxa.isEmpty(), "Expected no taxa to be recognized");
	}

	@Test
	public void testSingleAnnotationsSameView() throws Exception {
		addTaxonResolver(null);

		JCas jcas = JCasFactory.createJCas();
		jcas.setDocumentLanguage("en");
		jcas.setDocumentText("Ammophila arenaria");

		// manually insert a taxon annotation for "Ammophila arenaria"
		Taxon taxon = new Taxon(jcas, 0, 18);
		taxon.addToIndexes();
		AnnotationComment comment = new AnnotationComment(jcas);
		comment.setReference(taxon);
		comment.setKey("linking");
		comment.setValue("GBIF:1347914");
		comment.addToIndexes();

		composer.run(jcas);

		storeCas(jcas, "single_same_view");

		Collection<RecognizedTaxon> taxa = JCasUtil.select(jcas, RecognizedTaxon.class);
		assertEquals(1, taxa.size(), "Expected exactly one taxon to be recognized");
		RecognizedTaxon recognized = taxa.iterator().next();
		assertEquals("Ammophila arenaria", recognized.getText(), "Expected taxon text to match");
		assertEquals(1, recognized.getResolutions().size(), "Expected exactly one resolution");
		assertInstanceOf(TaxonResolution.class, recognized.getResolutions(0), "Expected resolution to be of type TaxonResolution");
		TaxonResolution resolution = (TaxonResolution) recognized.getResolutions(0);
		assertEquals("gbif", resolution.getProvider(), "Expected provider to be gbif");
		assertEquals("SCIENTIFIC", resolution.getNameType(), "Expected name type to be SCIENTIFIC");
		assertEquals("SPECIES", resolution.getRank(), "Expected rank to be SPECIES");
		assertEquals("Animalia", resolution.getKingdomName(), "Expected kingdom name to be 'Animalia'");
		assertEquals(1, resolution.getKingdomId(), "Expected kingdom ID to be 1");
		assertEquals("Hymenoptera", resolution.getOrderName(), "Expected order name to be 'Hymenoptera'");
		assertEquals(1457, resolution.getOrderId(), "Expected order ID to be 1457");
		assertEquals("Sphecidae", resolution.getFamilyName(), "Expected family name to be 'Sphecidae'");
		assertEquals(4352, resolution.getFamilyId(), "Expected family ID to be 4352");
		assertEquals("Podalonia", resolution.getGenusName(), "Expected genus name to be 'Podalonia'");
		assertEquals(1347780, resolution.getGenusId(), "Expected genus ID to be 1347780");
		assertEquals("Podalonia hirsuta", resolution.getSpeciesName(), "Expected species name to be 'Podalonia hirsuta'");
		assertEquals(1347914, resolution.getSpeciesId(), "Expected species ID to be 1347914");
		assertEquals("Podalonia hirsuta (Scopoli, 1763)", resolution.getScientificName(), "Expected scientific name to be 'Podalonia hirsuta (Scopoli, 1763)'");
	}

	@Test
	public void testSingleAnnotationsDifferentView() throws Exception {
		String annotationsView = "taxons";
		addTaxonResolver(annotationsView);

		JCas jcas = JCasFactory.createJCas();
		jcas.setDocumentLanguage("en");
		jcas.setDocumentText("Ammophila arenaria");

		// manually insert a taxon annotation for "Ammophila arenaria" in the specified view
		JCas annotationsJCas = jcas.createView(annotationsView);
		Taxon taxon = new Taxon(annotationsJCas, 0, 18);
		taxon.addToIndexes();
		AnnotationComment comment = new AnnotationComment(annotationsJCas);
		comment.setReference(taxon);
		comment.setKey("linking");
		comment.setValue("GBIF:1347914");
		comment.addToIndexes();

		composer.run(jcas);

		storeCas(jcas, "single_different_view");

		Collection<RecognizedTaxon> taxa = JCasUtil.select(jcas, RecognizedTaxon.class);
		assertEquals(1, taxa.size(), "Expected exactly one taxon to be recognized");
		RecognizedTaxon recognized = taxa.iterator().next();
		assertEquals("Ammophila arenaria", recognized.getText(), "Expected taxon text to match");
		assertEquals(1, recognized.getResolutions().size(), "Expected exactly one resolution");
		assertInstanceOf(TaxonResolution.class, recognized.getResolutions(0), "Expected resolution to be of type TaxonResolution");
		TaxonResolution resolution = (TaxonResolution) recognized.getResolutions(0);
		assertEquals("gbif", resolution.getProvider(), "Expected provider to be gbif");
		assertEquals(1347914, resolution.getSpeciesId(), "Expected species ID to be 1347914");
	}

	@Test
	public void testMultipleAnnotationsDifferentProviders() throws Exception {
		addTaxonResolver(null);

		JCas jcas = JCasFactory.createJCas();
		jcas.setDocumentLanguage("en");
		jcas.setDocumentText("Ammophila arenaria");

		// manually insert taxon annotations for "Ammophila arenaria" and "Ulex europaeus"
		Taxon taxon = new Taxon(jcas, 0, 18);
		taxon.addToIndexes();

		AnnotationComment comment1 = new AnnotationComment(jcas);
		comment1.setReference(taxon);
		comment1.setKey("linking");
		comment1.setValue("GBIF:1347914");
		comment1.addToIndexes();

		AnnotationComment comment2 = new AnnotationComment(jcas);
		comment2.setReference(taxon);
		comment2.setKey("linking");
		comment2.setValue("NCBI:96047");
		comment2.addToIndexes();

		AnnotationComment comment3 = new AnnotationComment(jcas);
		comment3.setReference(taxon);
		comment3.setKey("linking");
		comment3.setValue("TAXREF:82139");
		comment3.addToIndexes();

		composer.run(jcas);

		storeCas(jcas, "multiple_different_providers");

		Collection<RecognizedTaxon> taxa = JCasUtil.select(jcas, RecognizedTaxon.class);
		assertEquals(1, taxa.size(), "Expected exactly one taxon to be recognized");
		RecognizedTaxon recognized = taxa.iterator().next();
		assertEquals("Ammophila arenaria", recognized.getText(), "Expected taxon text to match");
		assertEquals(3, recognized.getResolutions().size(), "Expected exactly three resolutions");
		boolean hasGbif = false;
		boolean hasNcbi = false;
		boolean hasTaxref = false;
		for (int i = 0; i < recognized.getResolutions().size(); i++) {
			assertInstanceOf(TaxonResolution.class, recognized.getResolutions(i), "Expected resolution to be of type TaxonResolution");
			TaxonResolution resolution = (TaxonResolution) recognized.getResolutions(i);
			switch (resolution.getProvider()) {
				case "gbif" -> {
					hasGbif = true;
					assertEquals(1347914, resolution.getSpeciesId(), "Expected species ID to be 1347914 for GBIF");
				}
				case "ncbi" -> {
					hasNcbi = true;
					assertEquals(96047, resolution.getTaxonId(), "Expected taxon ID to be 96047 for NCBI");
					assertEquals("Calamagrostis", resolution.getParentName(), "Expected parent name to be 'Calamagrostis' for NCBI");
					assertEquals(15376, resolution.getParentId(), "Expected parent ID to be 15376 for NCBI");
					assertEquals("Calamagrostis arenaria", resolution.getScientificName(), "Expected scientific name to be 'Calamagrostis arenaria' for NCBI");
					assertEquals("species", resolution.getRank(), "Expected rank to be 'species' for NCBI");
				}
				case "taxref" -> {
					hasTaxref = true;
					assertEquals(82139, resolution.getTaxonId(), "Expected taxon ID to be 82139 for TAXREF");
					assertEquals("Plantae", resolution.getKingdomName(), "Expected kingdom name to be 'Plantae' for TAXREF");
					assertEquals("Equisetopsida", resolution.getClassName(), "Expected class name to be 'Equisetopsida' for TAXREF");
					assertEquals("Poales", resolution.getOrderName(), "Expected order name to be 'Poales' for TAXREF");
					assertEquals("Poaceae", resolution.getFamilyName(), "Expected family name to be 'Poaceae' for TAXREF");
					assertEquals("Pooideae", resolution.getSubfamilyName(), "Expected subfamily name to be 'Pooideae' for TAXREF");
					assertEquals("Poeae", resolution.getTribeName(), "Expected tribe name to be 'Poeae' for TAXREF");
					assertEquals("Ammophila arenaria", resolution.getSpeciesName(), "Expected species name to be 'Ammophila arenaria' for TAXREF");
					assertEquals("Ammophila arenaria", resolution.getScientificName(), "Expected scientific name to be 'Ammophila arenaria' for TAXREF");
					assertEquals("Oyat des sables, Ammophile des sables, Oyat, Chiendent marin, Roseau des sables, Gourbet", resolution.getVernacularName(), "Expected vernacular name to match for TAXREF");
					assertEquals("Ammophila arenaria (L.) Link, 1827", resolution.getAcceptedNameUsage(), "Expected accepted name usage to be 'Ammophila arenaria (L.) Link, 1827' for TAXREF");
					assertEquals("(L.) Link, 1827", resolution.getAuthorship(), "Expected authorship to be '(L.) Link, 1827' for TAXREF");
					assertEquals("species", resolution.getRank(), "Expected rank to be 'species' for TAXREF");
					assertEquals("https://taxref.mnhn.fr/taxref-web/taxa/82139", resolution.getReferences(), "Expeceted references to be 'https://taxref.mnhn.fr/taxref-web/taxa/82139' for TAXREF");
				}
				default -> fail("Unexpected provider: " + resolution.getProvider());
			}
		}
		assertTrue(hasGbif, "Expected a resolution from GBIF");
		assertTrue(hasNcbi, "Expected a resolution from NCBI");
		assertTrue(hasTaxref, "Expected a resolution from TAXREF");
	}

	static void storeCas(JCas cas, String name) {
		if (!STORE_OUTPUT) {
			return;
		}
		Path folderPath = Path.of(".", "outputs");
		try {
			Files.createDirectories(folderPath);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		Path path = folderPath.resolve(name + ".xmi");
		try (OutputStream os = Files.newOutputStream(path)) {
			XmiCasSerializer.serialize(cas.getCas(), os);
			System.out.println("CAS stored at: " + path.toAbsolutePath());
		} catch (Exception e) {
			System.err.println("Error serializing CAS: " + e.getMessage());
		}
	}

}

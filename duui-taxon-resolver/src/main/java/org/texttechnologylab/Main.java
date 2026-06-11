package org.texttechnologylab;

import java.io.InputStream;
import java.util.*;
import java.util.stream.Collectors;

import org.apache.uima.cas.CAS;
import org.apache.uima.cas.impl.XmiCasDeserializer;
import org.apache.uima.fit.factory.JCasFactory;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.Taxon;

public class Main {
	public static void main(String[] args) throws Exception {
		JCas jcas = JCasFactory.createJCas();
		try (InputStream is = ClassLoader.getSystemClassLoader().getResourceAsStream("4958866.xmi")) {
			CAS cas = jcas.getCas();
			XmiCasDeserializer.deserialize(is, cas);
		} catch (Exception e) {
			throw new Exception("Error deserializing CAS: " + e.getMessage());
		}
		JCas taxonView = jcas.getView("taxon");
		String documentText = jcas.getDocumentText();
		List<TaxonExtractor.RecognizedTaxon> recognizedTaxa = TaxonExtractor.extractTaxa(taxonView, documentText);
		System.out.println("Recognized taxa:");
		for (TaxonExtractor.RecognizedTaxon rt : recognizedTaxa) {
			System.out.println(rt);
		}

		/*
		NCBI - Resolved by Taxoniq
		GBIF - Resolved with API, page url: https://www.gbif.org/species/{taxonId}
		TAXREF - Downloaded backbone from https://ipt.gbif.fr/resource?r=taxref (Data as a DwC-A file)
		tsv file, use pandas for parsing
		 */
	}

	static String formatComment(AnnotationComment ac) {
		return "%s=%s".formatted(ac.getKey(), ac.getValue());
	}
}

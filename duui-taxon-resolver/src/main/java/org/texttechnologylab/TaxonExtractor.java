package org.texttechnologylab;

import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.texttechnologylab.annotation.AnnotationComment;
import org.texttechnologylab.annotation.type.Taxon;

import java.util.*;
import java.util.stream.Collectors;

public class TaxonExtractor {

	private TaxonExtractor() {
		// Utility class
		throw new UnsupportedOperationException("Utility class");
	}

	public static List<RecognizedTaxon> extractTaxa(JCas cas, String documentText) {
		Collection<AnnotationComment> annotationComments = JCasUtil.select(cas, AnnotationComment.class)
			.stream()
			.filter(ac -> ac.getReference() instanceof Taxon)
			.toList();
		Map<Integer, CasTaxon> casTaxa = new HashMap<>();
		for (AnnotationComment ac : annotationComments) {
			Taxon taxon = (Taxon) ac.getReference();
			int begin = taxon.getBegin();
			int end = taxon.getEnd();
			String text = documentText.substring(begin, end);
			CasTaxon ct = casTaxa.get(begin);
			if (ct == null) {
				casTaxa.put(begin, ct = new CasTaxon(begin, end, text));
			} else if (ct.end != end) {
				throw new RuntimeException("Overlapping taxa at position " + begin + ": " + ct.text + " and " + text);
			}
			int tid = taxon.getAddress();
			TaxonInfo info = ct.info.get(tid);
			if (info == null) {
				ct.info.put(tid, info = new TaxonInfo(taxon));
			}
			info.comments.add(ac);
		}

		return casTaxa.values().stream()
			.map(CasTaxon::toRecognizedTaxon)
			.toList();
	}

	public static class RecognizedTaxon {
		public final int begin, end;
		public final String text;
		public Map<String, List<Integer>> linkings = new HashMap<>();

		public RecognizedTaxon(int begin, int end, String text) {
			this.begin = begin;
			this.end = end;
			this.text = text;
		}

		@Override
		public String toString() {
			return "RecognizedTaxon{begin=%d, end=%d, text='%s', linkings=%s".formatted(begin, end, text, linkings);
		}
	}

	static class CasTaxon {
		final int begin, end;
		final String text;
		Map<Integer, TaxonInfo> info = new HashMap<>();

		public CasTaxon(int begin, int end, String text) {
			this.begin = begin;
			this.end = end;
			this.text = text;
		}

		@Override
		public String toString() {
			return "CasTaxon{begin=%d, end=%d, text='%s', info=%s}".formatted(begin, end, text, info);
		}

		RecognizedTaxon toRecognizedTaxon() {
			RecognizedTaxon rt = new RecognizedTaxon(begin, end, text);
			rt.linkings = info.values().stream()
				.flatMap(ti -> ti.comments.stream())
				.filter(ac -> "linking".equals(ac.getKey()))
				.map(AnnotationComment::getValue)
				.map(
					v -> {
						String[] parts = v.split(":");
						if (parts.length != 2) {
							throw new RuntimeException("Invalid linking value: " + v);
						}
						String source = parts[0];
						int id;
						try {
							id = Integer.parseInt(parts[1]);
						} catch (NumberFormatException e) {
							throw new RuntimeException("Invalid linking ID: " + parts[1]);
						}
						return Map.entry(source, id);
					}
				)
				.collect(
					Collectors.groupingBy(
						Map.Entry::getKey,
						Collectors.mapping(Map.Entry::getValue, Collectors.toList())
					)
				);
			return rt;
		}
	}

	static class TaxonInfo {
		final Taxon taxon;
		List<AnnotationComment> comments = new ArrayList<>();

		public TaxonInfo(Taxon taxon) {
			this.taxon = taxon;
		}

		@Override
		public String toString() {
			return "TaxonInfo{comments=%s}".formatted(comments.stream().map(Main::formatComment).toList());
		}
	}
}

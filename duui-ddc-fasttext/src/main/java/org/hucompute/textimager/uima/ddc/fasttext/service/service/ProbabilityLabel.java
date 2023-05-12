package org.hucompute.textimager.uima.ddc.fasttext.service.service;

public class ProbabilityLabel {
    private final String label;
    private final double logProb;
    private final double prob;

    ProbabilityLabel(String label, double logProb) {
        this.label = label;
        this.logProb = logProb;
        this.prob = Math.exp(this.logProb);
    }

    public String getLabel() {
        return label;
    }

    public double getLogProb() {
        return logProb;
    }

    public double getProb() {
        return prob;
    }
}

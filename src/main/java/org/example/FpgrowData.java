package org.example;

import java.util.List;

public class FpgrowData {
    private int consequent;
    private List<Integer> antecedent;
    private Double confidence;
    public FpgrowData(){}

    public FpgrowData(int consequent, List<Integer> antecedent, Double confidence) {
        this.consequent = consequent;
        this.antecedent = antecedent;
        this.confidence = confidence;
    }
    public int getConsequent() {
        return this.consequent;
    }
    public List<Integer> getAntecedent() {
        return this.antecedent;
    }
    public Double getConfidence() {
        return this.confidence;
    }
    public void setConsequent(int consequent) {
        this.consequent = consequent;
    }
    public void setAntecedent(List<Integer> antecedent) {
        this.antecedent = antecedent;
    }
    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }
}

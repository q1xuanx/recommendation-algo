package org.example;


import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class RecommendModel implements Serializable {
    private Map<Pair<String, Double>, Map<String, Map<String,Double>>> model;
    public void train (List<String> data){
        this.model = new HashMap<>();
        Map<String, Double> OD = new HashMap<>();
        Map<String, Map<String, Double>> CD = new HashMap<>();
        for (String m : data){
            List<String> foods = Arrays.stream(m.split(", ")).map(String::trim).collect(Collectors.toList());
            for (String f : foods){
                OD.putIfAbsent(f, 0.0);
                CD.putIfAbsent(f, new HashMap<>());
                OD.put(f, OD.get(f) + 1);
                for (String food : foods) {
                    if (!food.equals(f)) {
                        if (!CD.get(f).containsKey(food)) {
                            CD.get(f).put(food, 0.0);
                        }
                        CD.get(f).put(food, CD.get(f).get(food) + 1);
                    }
                }
            }
        }
        for (Map.Entry<String, Double> entry : OD.entrySet()){
            Pair<String, Double> mainPair = new Pair<>(entry.getKey(), entry.getValue());
            model.put(mainPair, new HashMap<>());
            model.get(mainPair).put(entry.getKey(), CD.get(entry.getKey()));
        }
    }
    public void saveModel(String filePath) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(Paths.get(filePath)))) {
            oos.writeObject(this.model);
        }
    }

    // Phương thức tải mô hình
    public void loadModel(InputStream modelStream ) throws IOException, ClassNotFoundException {
        try (ObjectInputStream ois = new ObjectInputStream(modelStream)) {
            this.model = (Map<Pair<String, Double>, Map<String, Map<String, Double>>>) ois.readObject();
        }
    }
    public Map<String, Double> recommend(List<String> data){
        Map<String, Double> recommendList = new HashMap<>();
        Map<String, List<Double>> P = new HashMap<>();
        Map<String, List<Double>> W = new HashMap<>();
        Map<String,Double> OD = new HashMap<>();
        Map<String, Map<String, Double>> CD = new HashMap<>();
        for (Map.Entry<Pair<String, Double>, Map<String,Map<String,Double>>> entry : this.model.entrySet()){
            OD.put(entry.getKey().getKey(), entry.getKey().getValue());
            CD.put(entry.getKey().getKey(), entry.getValue().get(entry.getKey().getKey()));
        }
        for (String inf : data){
            if (CD.get(inf) != null) {
                for (Map.Entry<String, Double> entry : CD.get(inf).entrySet()) {
                    if (!data.contains(entry.getKey())) {
                        P.putIfAbsent(entry.getKey(), new ArrayList<>());
                        W.putIfAbsent(entry.getKey(), new ArrayList<>());
                        Double p = CD.get(inf).get(entry.getKey()) / OD.get(inf);
                        P.get(entry.getKey()).add(p);
                        W.get(entry.getKey()).add(OD.get(inf));
                    }
                }
            }
        }
        for (String f : P.keySet()){
            double pSum = P.get(f).stream().mapToDouble(Double::doubleValue).sum();
            double wSum = W.get(f).stream().mapToDouble(Double::doubleValue).sum();
            recommendList.put(f, pSum * wSum);
        }
        return recommendList;
    }
}

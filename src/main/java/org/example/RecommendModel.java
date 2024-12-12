package org.example;
import java.io.*;
import java.util.*;
public class RecommendModel implements Serializable {
    public Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> trainPairwise (List<List<Integer>> data) throws InterruptedException {
        Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> tm;
        Map<Integer, Double> OD = new HashMap<>();
        Map<Integer, Map<Integer, Double>> CD = new HashMap<>();
        for (List<Integer> foods : data) {
            for (Integer f : foods) {
                if (OD.containsKey(f)){
                    OD.put(f, OD.get(f) + 1);
                }else {
                    OD.put(f, 1.0);
                    CD.put(f, new HashMap<>());
                }
                for (Integer food : foods) {
                    if (!food.equals(f)) {
                        if (!CD.get(f).containsKey(food)) {
                            CD.get(f).put(food, 0.0);
                        }
                        CD.get(f).put(food, CD.get(f).get(food) + 1);
                    }
                }
            }
        }
        tm = constructData(OD,CD);
        return tm;
    }

    public Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> constructData(Map<Integer,Double> OD, Map<Integer, Map<Integer,Double>> CD) throws InterruptedException {
        Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> tm = new HashMap<>();
        for (Map.Entry<Integer, Double> entry : OD.entrySet()){
            Pair<Integer, Double> mainPair = new Pair<>(entry.getKey(), entry.getValue());
            tm.put(mainPair, new HashMap<>());
            tm.get(mainPair).put(entry.getKey(), CD.get(entry.getKey()));
        }
        return tm;
    }

    public Map<Integer, Double> recommendPairwise (Map<Pair<Integer, Double>, Map<Integer,Map<Integer,Double>>> model, List<Integer> data, float recommend_score){
        Map<Integer, Double> recommendList = new HashMap<>();
        Map<Integer, List<Double>> P = new HashMap<>();
        Map<Integer, List<Double>> W = new HashMap<>();
        Map<Integer,Double> OD = new HashMap<>();
        Map<Integer, Map<Integer, Double>> CD = new HashMap<>();
        for (Map.Entry<Pair<Integer, Double>, Map<Integer,Map<Integer,Double>>> entry : model.entrySet()){
            OD.put(entry.getKey().getKey(), entry.getKey().getValue());
            CD.put(entry.getKey().getKey(), entry.getValue().get(entry.getKey().getKey()));
        }
        for (Integer inf : data){
            if (CD.get(inf) != null) {
                for (Map.Entry<Integer, Double> entry : CD.get(inf).entrySet()) {
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
        for (Integer f : P.keySet()){
            double pSum = P.get(f).stream().mapToDouble(Double::doubleValue).sum();
            double wSum = W.get(f).stream().mapToDouble(Double::doubleValue).sum();
            recommendList.put(f, pSum * wSum);
        }
        recommendList.entrySet().removeIf(entry -> entry.getValue() < recommend_score);
        return recommendList;
    }

    public Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> trainConstraint(List<List<Integer>> data, List<Integer> InputFood) throws InterruptedException {
        Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> tm = new HashMap<>();
        Map<List<Integer>, Double> OD = new HashMap<>();
        Map<List<Integer>, Map<Integer, Double>> CD = new HashMap<>();
        Set<Integer> ifFood = new HashSet<>(InputFood);
        List<Integer> fFood = new ArrayList<>(ifFood);
        OD.put(fFood, 0.0);
        CD.put(fFood, new HashMap<>());
        for (int i = 0; i < data.size(); i++){
            List<Integer> foods = data.get(i);
            boolean isContains = new HashSet<>(foods).containsAll(fFood);
            if (isContains) {
                List<Integer> remain = new ArrayList<>(foods);
                remain.removeAll(fFood);
                OD.put(fFood, OD.get(fFood) + 1);
                for (Integer food : remain) {
                    if (!CD.get(fFood).containsKey(food)) {
                        CD.get(fFood).put(food, 1.0);
                    } else {
                        CD.get(fFood).put(food, CD.get(fFood).get(food) + 1);
                    }
                }
            }
        }
        for (Map.Entry<List<Integer>, Double> entry : OD.entrySet()){
            Pair<List<Integer>, Double> pair = new Pair<>(entry.getKey(), entry.getValue());
            tm.put(pair, new HashMap<>());
            tm.get(pair).put(entry.getKey(), CD.get(entry.getKey()));
        }
        return tm;
    }

    public Map<Integer,Double> recommendConstraint(Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> model){
        Map<Integer, Double> recommend = new HashMap<>();
        Map<Integer, List<Double>> P = new HashMap<>();
        Map<Integer, List<Double>> W = new HashMap<>();
        Map<List<Integer>, Double> OD = new HashMap<>();
        Map<List<Integer>, Map<Integer, Double>> CD = new HashMap<>();
        for (Map.Entry<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer, Double>>> entry : model.entrySet()){
            OD.put(entry.getKey().getKey(), entry.getKey().getValue());
            CD.put(entry.getKey().getKey(), entry.getValue().get(entry.getKey().getKey()));
        }
        for (Map.Entry<List<Integer>, Map<Integer, Double>> entry : CD.entrySet()){
            for (Map.Entry<Integer, Double> entry2 : entry.getValue().entrySet()){
                if (!P.containsKey(entry2.getKey())) {
                    P.put(entry2.getKey(), new ArrayList<>());
                    W.put(entry2.getKey(), new ArrayList<>());
                }
                Double p = CD.get(entry.getKey()).get(entry2.getKey()) / OD.get(entry.getKey());
                P.get(entry2.getKey()).add(p);
                W.get(entry2.getKey()).add(OD.get(entry.getKey()));
            }
        }
        for (Integer food : P.keySet()) {
            Double pSum = P.get(food).stream().mapToDouble(Double::doubleValue).sum();
            Double wSum = W.get(food).stream().mapToDouble(Double::doubleValue).sum();
            recommend.put(food, (pSum * wSum));
        }
        return recommend;
    }
}

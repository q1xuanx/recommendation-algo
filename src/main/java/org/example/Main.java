package org.example;

import javafx.util.Pair;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import scala.Char;

import java.sql.SQLOutput;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
//        SparkSession spark = SparkSession
//                .builder()
//                .master("local[*]")
//                .appName("SparkFPGrowth")
//                .getOrCreate();
//        List<Row> data = Arrays.asList(
//                RowFactory.create(Arrays.asList("a b c d".split(" "))),
//                RowFactory.create(Arrays.asList("a d e".split(" "))),
//                RowFactory.create(Arrays.asList("d e".split(" "))),
//                RowFactory.create(Arrays.asList("a b".split(" ")))
//                );
//        StructType schema = new StructType(new StructField[]{new StructField(
//                "items", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
//        });
//        Dataset<Row> itemsDF = spark.createDataFrame(data, schema);
//        FPGrowthModel model = new FPGrowth().setItemsCol("items").setMinSupport(0.25).setMinConfidence(0.3).fit(itemsDF);
//        model.associationRules().show();
        List<String> IF = new ArrayList<>();
        for (char i = 'a'; i < 'c'; i++){
            IF.add(String.valueOf(i));
        }
//        Map<String,Double> recommendList = algoRecommend1(model,IF);
//        System.out.println("Recommendation Algo 1: " + recommendList);
        List<String> data1 = new ArrayList<>();
        data1.add("a b c d");
        data1.add("a d e");
        data1.add("d e");
        data1.add("a b");
        Map<Integer, List<Pair<Character, Double>>> trainRes = Train1(data1);
        Map<Character, Double> recommendList1 = algoRecommend2(trainRes, IF);
        System.out.println("Train Algo 2: " + trainRes);
        System.out.println("Recommendation Algo 2: " + recommendList1);
    }
    //Algo 1 based Association rules
    public static Map<String,Double> algoRecommend1(FPGrowthModel model, List<String> IF){
        Dataset<Row> asRule = model.associationRules();
        Map<String, Double> recommendList = new HashMap<>();
        Set<String> IFSet = new HashSet<>(IF);
        for (Row row : asRule.collectAsList()){
            String f = String.valueOf(row.getList(1).toString().charAt(1));
            if (!IFSet.contains(f)){
                Set<String> antc = row.getList(0).stream().map(String::valueOf).collect(Collectors.toSet());
                Set<String> antcSet = new HashSet<>(antc);
                antcSet.retainAll(IFSet);
                long countSize = antcSet.size();
                if (countSize > 0){
                    if (!recommendList.containsKey(f)){
                        recommendList.put(f, 0.0);
                    }
                    double c = row.getDouble(2);
                    double ms = ((double) (countSize * countSize) / (antc.size() * IF.size()));
                    recommendList.put(f, recommendList.get(f) + c * ms);
                }
            }
        }
        return recommendList;
    }
    //Algo 2 based transactional item confidence;
    public static Map<Integer, List<Pair<Character, Double>>> Train1 (List<String> data){
        Map<Integer, List<Pair<Character, Double>>> tm = new HashMap<>();
        data.replaceAll(s -> s.replace(" ", ""));
        for (int i = 0; i < data.size(); i++){
            if (!tm.containsKey(i)) {
                tm.put(i, new ArrayList<>());
            }
            String m = data.get(i);
            List<Pair<Character, Double>> saveConf = new ArrayList<>();
            int cm = (int) data.stream().filter(s -> s.contains(m)).count();
            for (int j = 0; j < m.length(); j++){
                char f = m.charAt(j);
                int cf = getCf(data, m, j);
                Pair<Character, Double> pair = new Pair<>(f, (double) cm / cf);
                saveConf.add(pair);
            }
            tm.put(i, new ArrayList<>(saveConf));
        }
        return tm;
    }
    private static int getCf(List<String> data, String m, int j) {
        StringBuilder tm2 = new StringBuilder();
        for (int z = 0; z < m.length(); z++){
            if (z == j) continue;
            tm2.append(m.charAt(z));
        }
        int cf = 0;
        for (String datum : data){
            boolean isContains = true;
            for (int z = 0; z < tm2.length(); z++){
                if (!datum.contains(String.valueOf(tm2.charAt(z)))){
                    isContains = false;
                    break;
                }
            }
            if (isContains){
                cf++;
            }
        }
        return cf;
    }
    public static Map<Character, Double> algoRecommend2(Map<Integer, List<Pair<Character, Double>>> model, List<String> data){
        Map<Character,Double> recommendList = new HashMap<>();
        for (Map.Entry<Integer, List<Pair<Character, Double>>> entry : model.entrySet()){
            List<Pair<Character, Double>> tempList = new ArrayList<>(entry.getValue());
            int f2 = 0;
            System.out.println(tempList);
            for (String charFound : data) {
                boolean isContains = tempList.removeIf(s -> s.getKey().toString().equals(charFound));
                if (isContains) f2++;
            }
            if (f2 != 0){
                for (Pair<Character, Double> characterDoublePair : tempList) {
                    if (!recommendList.containsKey(characterDoublePair.getKey())) {
                        recommendList.put(characterDoublePair.getKey(), 0.0);
                    }
                    double conf = characterDoublePair.getValue();
                    recommendList.put(characterDoublePair.getKey(), recommendList.get(characterDoublePair.getKey()) + (double) f2 * conf);
                }
            }
        }
        return recommendList;
    }
}
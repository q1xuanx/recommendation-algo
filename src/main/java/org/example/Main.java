package org.example;

import au.com.bytecode.opencsv.CSVReader;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.sources.In;
import org.apache.spark.sql.types.*;
import scala.Char;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.sql.SQLOutput;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
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
        //model.associationRules().show();
        List<String> data = new ArrayList<>();
        data.add("a, b, c, d");
        data.add("a, d, e");
        data.add("d, e");
        data.add("a, b");
        List<String> IF = new ArrayList<>();
//        for (char i = 'a'; i < 'c'; i++){
//            IF.add(String.valueOf(i));
//        }
        IF.add("Tôm Sốt Cà");
        IF.add("Chuối Già");
//        IF.add("Cơm");
//        IF.add("Gà Xào Rau Củ");
//        Map<String,Double> recommendList = algoRecommend1(model,IF);
//        System.out.println("-> Food recommend with algo 1: ");
//        for (Map.Entry<String, Double> entry : recommendList.entrySet()) {
//            System.out.println(entry);
//        }
        List<String> data1 = readData("C:\\Users\\ADMIN\\Desktop\\AsRecomenResearch\\research\\src\\main\\java\\org\\example\\dataset.csv");
        System.out.println("Pairwise default: ");
        Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainSet = Train2(data1);
        System.out.println("Recommend Pairwise: ");
        Map<String,Double> rec1 = algoRecommend3(trainSet, IF);
        System.out.println(rec1);
        System.out.println("Pairwise constrain leased");
        Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainSet2 = trainConstraint(data1, IF);
        System.out.println("Constraint as rules: ");
        Map<String,Double> rec2 = recommendConstraint(trainSet2);
        System.out.println(rec2);
        //        recommendModel.saveModel("/modelRecommend.ser");
//        System.out.println("Save Success");
        //setOfData(data1);
//        InputStream modelStream = Main.class.getResourceAsStream("/modelRecommend.ser");
//        recommendModel.loadModel(modelStream);
//        for (Map.Entry<String, Double> entry : recommendModel.recommend(IF).entrySet()){
//            System.out.println(entry.getKey() + ": " + entry.getValue());
//        }
    }
    public static void printData(Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainSet){
        for (Map.Entry<Pair<String, Double>, Map<String, Map<String,Double>>> entry : trainSet.entrySet()) {
            System.out.println(entry.getKey().getKey() + " " + entry.getKey().getValue());
            for (Map.Entry<String, Map<String,Double>> entry1 : entry.getValue().entrySet()) {
                System.out.println(entry1.getKey() + " " + entry1.getValue());
            }
        }
    }
    public static List<String> readData(String nameDataSet){
        List<String> data = new ArrayList<>();
        List<List<String>> records = new ArrayList<>();
        try(CSVReader csvReader = new CSVReader(new FileReader(nameDataSet));){
            String[] nextRecord = null;
            while ((nextRecord = csvReader.readNext()) != null){
                records.add(Arrays.asList(nextRecord));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        for(int i = 1; i < records.size(); i++){
            data.add(records.get(i).get(1));
        }
        return data;
    }
    public static void saveModel(Map<Pair<String, Double>, Map<String,Map<String,Double>>> model, String filePath){
        try(ObjectOutputStream oos = new ObjectOutputStream(Files.newOutputStream(Paths.get(filePath)))){
            oos.writeObject(model);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
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
    public static Map<Integer, List<Pair<String, Double>>> Train1 (List<String> data){
        Map<Integer, List<Pair<String, Double>>> tm = new HashMap<>();
        for (int i = 0; i < data.size(); i++){
            if (!tm.containsKey(i)) {
                tm.put(i, new ArrayList<>());
            }
            String m = data.get(i);
            List<Pair<String, Double>> saveConf = new ArrayList<>();
            int cm = (int) data.stream().filter(s -> s.contains(m)).count();
            List<String> datas = Arrays.stream(m.split(", ")).map(String::trim).collect(Collectors.toList());
            for (int j = 0; j < datas.size(); j++){
                String f = datas.get(j);
                int cf = getCf(data, datas, j);
                Pair<String, Double> pair = new Pair<>(f, (double) cm / cf);
                saveConf.add(pair);
            }
            tm.put(i, new ArrayList<>(saveConf));
        }
        return tm;
    }
    private static int getCf(List<String> data, List<String> datas, int j) {
        StringBuilder tm2 = new StringBuilder();
        for (int z = 0; z < datas.size(); z++){
            if (z == j) continue;
            tm2.append(datas.get(z)).append(" ");
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

    public static Map<String, Double> algoRecommend2(Map<Integer, List<Pair<String, Double>>> model, List<String> data){
        Map<String,Double> recommendList = new HashMap<>();
        for (Map.Entry<Integer, List<Pair<String, Double>>> entry : model.entrySet()){
            List<Pair<String, Double>> tempList = new ArrayList<>(entry.getValue());
            int f2 = 0;
            for (String charFound : data) {
                boolean isContains = tempList.removeIf(s -> s.getKey().equals(charFound));
                if (isContains) f2++;
            }
            if (f2 != 0){
                for (Pair<String, Double> characterDoublePair : tempList) {
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
    public static void setOfData(List<String> data){
        Set<String> set = new HashSet<>();
        for (String m : data){
            List<String> foods = Arrays.stream(m.split(", ")).map(String::trim).collect(Collectors.toList());
            set.addAll(foods);
        }
        for (String s : set){
            System.out.println("\"" + s + "\", ");
        }
    }
    //Algo 3 Train based pair wise associations rules
    public static Map<Pair<String, Double>, Map<String, Map<String,Double>>> Train2 (List<String> data){
        Map<Pair<String, Double>, Map<String, Map<String,Double>>> tm = new HashMap<>();
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
        tm = constructData(OD,CD);
        return tm;
    }
    public static Map<Pair<String, Double>, Map<String, Map<String,Double>>> constructData(Map<String,Double> OD, Map<String, Map<String,Double>> CD){
        Map<Pair<String, Double>, Map<String, Map<String,Double>>> tm = new HashMap<>();
        for (Map.Entry<String, Double> entry : OD.entrySet()){
            Pair<String, Double> mainPair = new Pair<>(entry.getKey(), entry.getValue());
            tm.put(mainPair, new HashMap<>());
            tm.get(mainPair).put(entry.getKey(), CD.get(entry.getKey()));
        }
        return tm;
    }
    //Constrain leased associations rules
    public static Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainConstraint(List<String> data, List<String> InputFood){
        Map<Pair<String, Double>, Map<String, Map<String,Double>>> tm;
        Map<String, Double> OD = new HashMap<>();
        Map<String, Map<String, Double>> CD = new HashMap<>();
        StringBuilder builder = new StringBuilder();
        Set<String> ifFood = new HashSet<>(InputFood);
        List<String> fFood = new ArrayList<>(ifFood);
        for(String m: fFood){
            builder.append(m).append(" ");
        }
        String query = builder.toString().trim();
        OD.put(query, 0.0);
        CD.put(query, new HashMap<>());
        for (String m : data){
            List<String> foods  = Arrays.stream(m.split(", ")).map(String::trim).collect(Collectors.toList());
            boolean isContains = new HashSet<>(foods).containsAll(fFood);
            if (isContains) {
                List<String> remain = new ArrayList<>(foods);
                remain.removeAll(fFood);
                OD.put(query, OD.get(query) + 1);
                for (String food : remain) {
                    if (!CD.get(query).containsKey(food)) {
                        CD.get(query).put(food, 1.0);
                    } else {
                        CD.get(query).put(food, CD.get(query).get(food) + 1);
                    }
                }
            }
        }
        tm = constructData(OD,CD);
        return tm;
    }
    public static Map<String,Double> recommendConstraint(Map<Pair<String, Double>, Map<String,Map<String,Double>>> model){
        Map<String, Double> recommend = new HashMap<>();
        Map<String, List<Double>> P = new HashMap<>();
        Map<String, List<Double>> W = new HashMap<>();
        Map<String,Double> OD = new HashMap<>();
        Map<String, Map<String, Double>> CD = new HashMap<>();
        for (Map.Entry<Pair<String, Double>, Map<String,Map<String,Double>>> entry : model.entrySet()){
            OD.put(entry.getKey().getKey(), entry.getKey().getValue());
            CD.put(entry.getKey().getKey(), entry.getValue().get(entry.getKey().getKey()));
        }
        for (Map.Entry<String, Map<String,Double>> entry : CD.entrySet()){
            for (Map.Entry<String,Double> entry2 : entry.getValue().entrySet()){
                if (!P.containsKey(entry2.getKey())) {
                    P.put(entry2.getKey(), new ArrayList<>());
                    W.put(entry2.getKey(), new ArrayList<>());
                }
                Double p = CD.get(entry.getKey()).get(entry2.getKey()) / OD.get(entry.getKey());
                P.get(entry2.getKey()).add(p);
                W.get(entry2.getKey()).add(OD.get(entry.getKey()));
            }
        }
        for (String food : P.keySet()) {
            Double pSum = P.get(food).stream().mapToDouble(Double::doubleValue).sum();
            Double wSum = W.get(food).stream().mapToDouble(Double::doubleValue).sum();
            recommend.put(food, (pSum * wSum));
        }
        return recommend;
    }
    public static Map<String, Double> algoRecommend3(Map<Pair<String, Double>, Map<String,Map<String,Double>>> model, List<String> data){
        Map<String, Double> recommendList = new HashMap<>();
        Map<String, List<Double>> P = new HashMap<>();
        Map<String, List<Double>> W = new HashMap<>();
        Map<String,Double> OD = new HashMap<>();
        Map<String, Map<String, Double>> CD = new HashMap<>();
        for (Map.Entry<Pair<String, Double>, Map<String,Map<String,Double>>> entry : model.entrySet()){
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
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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        List<String> datainput = readData("C:\\Users\\ADMIN\\Desktop\\AsRecomenResearch\\research\\src\\main\\java\\org\\example\\dataset.csv");
        int iterations = 100;
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("SparkFPGrowth")
                .getOrCreate();
//        List<Row> data = Arrays.asList(
//                RowFactory.create(Arrays.asList("a b c d".split(" "))),
//                RowFactory.create(Arrays.asList("a d e".split(" "))),
//                RowFactory.create(Arrays.asList("d e".split(" "))),
//                RowFactory.create(Arrays.asList("a b".split(" ")))
//                );
        List<Row> data = new ArrayList<>();
        for (String s : datainput) {
            data.add(RowFactory.create(Arrays.asList(s.split(", "))));
        }
//        List<String> data1 = new ArrayList<>();
//        data1.add("a, b, c, d");
//        data1.add("a, d, e");
//        data1.add("d, e");
//        data1.add("a, b");
        Map<String, Integer> mappingData = prepareData(datainput);
        Map<Integer, String> reverseDataMapping = reverseDataMapping(mappingData);
        //System.out.println(data);

        //System.out.println(data);
        StructType schema = new StructType(new StructField[]{new StructField(
                "items", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> itemsDF = spark.createDataFrame(data, schema);
        FPGrowthModel model = new FPGrowth().setItemsCol("items").setMinSupport(0.01).setMinConfidence(0.6).fit(itemsDF);
//        long genARTime = runtimeCalculate(() -> {
//            FPGrowthModel model1 = new FPGrowth().setItemsCol("items").setMinSupport(0.01).setMinConfidence(0.1).fit(itemsDF);
//        }, iterations);
        //model.associationRules().show();
        List<String> IF = new ArrayList<>();
//        IF.add("a");
//        IF.add("b");
        IF.add("Cơm");
        IF.add("Chả Chay Kho Tiêu");
        List<Integer> dataIf = changeInputFoodToNum(IF, mappingData);
        List<FpgrowData> listData = prepareForFpModel(model, mappingData);
        List<List<Integer>> dataNumber = changeDataTrainToNumber(datainput, mappingData);
        Map<Integer, Double> recommend1 = algoRecommend1(listData, dataIf);
        System.out.println("-> Food recommend based Association Rules: ");
        for (Map.Entry<Integer, Double> en : recommend1.entrySet()){
            en.setValue(Math.round(en.getValue() * 100.0) / 100.0);
            System.out.println(reverseDataMapping.get(en.getKey()) + " " + en.getValue());
        }
//        long avgRecommend1 = runtimeCalculate(() -> {
//            Map<String,Double> recommendList1 = algoRecommend1(model,IF);
//        },iterations);
//        System.out.println("Avg gen AS rules: " + genARTime + "ms");
//        System.out.println("AVG time for AR: " + avgRecommend1 + "ms");
        //        for (Map.Entry<String, Double> entry : recommendList.entrySet()) {
//            entry.setValue(Math.round(entry.getValue() * 100.0) / 100.0);
//            System.out.println(entry);
//        }
        Map<Integer, List<Pair<Integer, Double>>> train1 = Train1(dataNumber);
//        float avgTrain1 = runtimeCalculate(() -> {
//            Map<Integer, List<Pair<String, Double>>> trainCal = Train1(datainput);
//        }, iterations);
        System.out.println("-> Food recommend based Transactional Item Confidence: ");
        //System.out.println("AVG train for TIC: " + avgTrain1 + "ms");
        Map<Integer, Double> rec1 = algoRecommend2(train1, dataIf);
//        float avgRec1 = runtimeCalculate(() -> {
//            Map<String, Double> recCal = algoRecommend2(train1, IF);
//        }, iterations);
//        System.out.println("AVG time for TIC: " + avgRec1 + "ms");
//        Map<String, Double> sorted = rec1.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed()).collect(LinkedHashMap::new, (m, e)-> m.put(e.getKey(), e.getValue()), LinkedHashMap::putAll);
        for (Map.Entry<Integer, Double> entry : rec1.entrySet()){
            entry.setValue(Math.round(entry.getValue() * 100.0) / 100.0);
            System.out.println(reverseDataMapping.get(entry.getKey()) + " " + entry.getValue());
        }
        Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> trainSet = Train2(dataNumber);
//        float avgTrain2 = runtimeCalculate(() -> {
//            try {
//                Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainCal = Train2(datainput);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//        }, iterations);
        System.out.println("-> Food recommend based Pairwise Association Rules: ");
        //System.out.println("AVG train for PAR: " + avgTrain2 + "ms");
       Map<Integer,Double> rec2 = algoRecommend3(trainSet, dataIf);
//        float avgRec2 = runtimeCalculate(() -> {
//            Map<String,Double> recCal = algoRecommend3(trainSet, IF);
//        }, iterations);
//        System.out.println("AVG time for PAR: " + avgRec2 + "ms");
        //Map<String,Double> sorted2 = rec2.entrySet().stream().sorted(Map.Entry.<String, Double>comparingByValue().reversed()).collect(LinkedHashMap::new, (m,e) -> m.put(e.getKey(), e.getValue()), LinkedHashMap::putAll);
        for (Map.Entry<Integer, Double> entry : rec2.entrySet()){
            entry.setValue(Math.round(entry.getValue() * 10.0) / 10.0);
            System.out.println(reverseDataMapping.get(entry.getKey()) + " " + entry.getValue());
        }
        Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> trainSet2 = trainConstraint(dataNumber, dataIf);
//        for (Map.Entry<Pair<String, Double>, Map<String, Map<String,Double>>> entry : trainSet.entrySet()){
//            System.out.print(entry.getKey().getKey() + " " + entry.getKey().getValue() + " => ");
//            for (Map.Entry<String, Map<String,Double>> entry1 : entry.getValue().entrySet()){
//                System.out.println(entry1.getValue());
//            }
//        }
//        float avgTrain3 = runtimeCalculate(() -> {
//            try {
//                Map<Pair<String, Double>, Map<String, Map<String,Double>>> trainCal = trainConstraint(datainput, IF);
//            } catch (InterruptedException e) {
//                throw new RuntimeException(e);
//            }
//        }, iterations);
        System.out.println("-> Food recommend based Constraint Association Rules: ");
        //System.out.println("AVG time train for CAR: " + avgTrain3 + "ms");
        Map<Integer,Double> rec3 = recommendConstraint(trainSet2);
//        float avgRec3 = runtimeCalculate(() -> {
//            Map<String,Double> recCal = recommendConstraint(trainSet2);
//        }, iterations);
//        System.out.println("AVG time for CAR: " + avgRec3 + "ms");
        for (Map.Entry<Integer, Double> entry : rec3.entrySet()){
            entry.setValue(Math.round(entry.getValue() * 100.0) / 100.0);
            System.out.println(reverseDataMapping.get(entry.getKey()) + " " + entry.getValue());
        }
        //        recommendModel.saveModel("/modelRecommend.ser");
//        System.out.println("Save Success");
        //setOfData(data1);
//        InputStream modelStream = Main.class.getResourceAsStream("/modelRecommend.ser");
//        recommendModel.loadModel(modelStream);
//        for (Map.Entry<String, Double> entry : recommendModel.recommend(IF).entrySet()){
//            System.out.println(entry.getKey() + ": " + entry.getValue());
//        }
    }
    public static float runtimeCalculate(Runnable task, int iterations){
        long totalTime = 0;
        for (int i = 0; i < iterations; i++) {
            long start = System.currentTimeMillis();
            task.run();
            long end = System.currentTimeMillis();
            totalTime += (end - start);
        }
        return (float) totalTime / iterations;
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
    public static Map<String, Integer> prepareData(List<String> data){
        int i = 1;
        Map<String, Integer> listDataToMapping = new HashMap<>();
        for (String s : data){
            List<String> split = Arrays.stream(s.split(", ")).map(String::valueOf).collect(Collectors.toList());
            for (String s1 : split){
                if (!listDataToMapping.containsKey(s1)){
                    listDataToMapping.put(s1, i);
                    i++;
                }
            }
        }
        return listDataToMapping;
    }
    public static List<List<Integer>> changeDataTrainToNumber(List<String> data, Map<String, Integer> mappingValue){
        List<List<Integer>> listData = new ArrayList<>();
        for (String s : data){
            String[] temp = s.split(", ");
            List<Integer> dataAdd = new ArrayList<>();
            for (String s2 : temp){
                dataAdd.add(mappingValue.get(s2));
            }
            listData.add(dataAdd);
        }
        return listData;
    }
    public static Map<Integer, String> reverseDataMapping(Map<String, Integer> data){
        Map<Integer, String> reverseData = new HashMap<>();
        for (Map.Entry<String, Integer> entry : data.entrySet()){
            reverseData.put(entry.getValue(), entry.getKey());
        }
        return reverseData;
    }
    public static List<Integer> changeInputFoodToNum(List<String> data, Map<String, Integer> dataForMapping){
        List<Integer> dataPrepared = new ArrayList<>();
        for (String s : data){
            dataPrepared.add(dataForMapping.get(s));
        }
        return dataPrepared;
    }
    public static List<FpgrowData> prepareForFpModel (FPGrowthModel model, Map<String, Integer> mappingValue){
        Dataset<Row> asRule = model.associationRules();
        List<FpgrowData> dataPrepared = new ArrayList<>();
        for (Row row : asRule.collectAsList()){
            String consequent = row.getList(1).toString().replace("[", "").replace("]", "");
            Set<String> antecedents = row.getList(0).stream().map(String::valueOf).collect(Collectors.toSet());
            List<Integer> tempAntecedents = new ArrayList<>();
            for (String antecedent : antecedents){
                tempAntecedents.add(mappingValue.get(antecedent));
            }
            dataPrepared.add(new FpgrowData(mappingValue.get(consequent), tempAntecedents, row.getDouble(2)));
        }
        return dataPrepared;
    }
    //Algo 1 based Association rules
    public static Map<Integer,Double> algoRecommend1(List<FpgrowData> listData, List<Integer> IF){
        Map<Integer, Double> recommendList = new HashMap<>();
        Set<Integer> IFSet = new HashSet<>(IF);
        for (FpgrowData fp : listData){
            int f = fp.getConsequent();
            if (!IFSet.contains(f)){
                Set<Integer> antc = new HashSet<>(fp.getAntecedent());
                Set<Integer> antcSet = new HashSet<>(antc);
                antcSet.retainAll(IFSet);
                long countSize = antcSet.size();
                if (countSize > 0){
                    if (!recommendList.containsKey(f)){
                        recommendList.put(f, 0.0);
                    }
                    double c = fp.getConfidence();
                    double ms = ((double) (countSize * countSize) / (antc.size() * IF.size()));
                    recommendList.put(f, recommendList.get(f) + c * ms);
                }
            }
        }
        return recommendList;
    }
    //Algo 2 based transactional item confidence;
    public static Map<Integer, List<Pair<Integer, Double>>> Train1 (List<List<Integer>> data){
        Map<Integer, List<Pair<Integer, Double>>> tm = new HashMap<>();
        for (int i = 0; i < data.size(); i++){
            if (!tm.containsKey(i)) {
                tm.put(i, new ArrayList<>());
            }
            List<Integer> m = data.get(i);
            List<Pair<Integer, Double>> saveConf = new ArrayList<>();
            int cm = (int) data.stream().filter(s -> Objects.equals(s, m)).count();
            for (int j = 0; j < m.size(); j++){
                Integer f = m.get(j);
                int cf = getCf(data, m, j);
                Pair<Integer, Double> pair = new Pair<>(f, (double) cm / cf);
                saveConf.add(pair);
            }
            tm.put(i, new ArrayList<>(saveConf));
        }
        return tm;
    }
    private static int getCf(List<List<Integer>> data, List<Integer> datas, int j) {
        int[] temp = new int[datas.size() - 1];
        int index = 0;
        for (int z = 0; z < datas.size(); z++){
            if (z == j) continue;
            temp[index] = datas.get(z);
            index++;
        }
        int cf = 0;
        for (List<Integer> datum : data){
            boolean isContains = true;
            for (int i : temp){
                if (!datum.contains(i)){
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

    public static Map<Integer, Double> algoRecommend2(Map<Integer, List<Pair<Integer, Double>>> model, List<Integer> data){
        Map<Integer,Double> recommendList = new HashMap<>();
        for (Map.Entry<Integer, List<Pair<Integer, Double>>> entry : model.entrySet()){
            List<Pair<Integer, Double>> tempList = new ArrayList<>(entry.getValue());
            int f2 = 0;
            for (Integer numFound : data) {
                boolean isContains = tempList.removeIf(s -> s.getKey().equals(numFound));
                if (isContains) f2++;
            }
            if (f2 != 0){
                for (Pair<Integer, Double> numberPair : tempList) {
                    if (!recommendList.containsKey(numberPair.getKey())) {
                        recommendList.put(numberPair.getKey(), 0.0);
                    }
                    double conf = numberPair.getValue();
                    recommendList.put(numberPair.getKey(), recommendList.get(numberPair.getKey()) + (double) f2 * conf);
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
    public static Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> Train2 (List<List<Integer>> data) throws InterruptedException {
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
    public static Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> constructData(Map<Integer,Double> OD, Map<Integer, Map<Integer,Double>> CD) throws InterruptedException {
        Map<Pair<Integer, Double>, Map<Integer, Map<Integer,Double>>> tm = new HashMap<>();
        for (Map.Entry<Integer, Double> entry : OD.entrySet()){
            Pair<Integer, Double> mainPair = new Pair<>(entry.getKey(), entry.getValue());
            tm.put(mainPair, new HashMap<>());
            tm.get(mainPair).put(entry.getKey(), CD.get(entry.getKey()));
        }
        return tm;
    }
    public static Map<Integer, Double> algoRecommend3(Map<Pair<Integer, Double>, Map<Integer,Map<Integer,Double>>> model, List<Integer> data){
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
        return recommendList;
    }
    //Constrain leased associations rules
    public static Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> trainConstraint(List<List<Integer>> data, List<Integer> InputFood) throws InterruptedException {
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
    public static Map<Integer,Double> recommendConstraint(Map<Pair<List<Integer>, Double>, Map<List<Integer>, Map<Integer,Double>>> model){
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
package org.example;

import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.fpm.FPGrowthModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;

import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        SparkSession spark = SparkSession
                .builder()
                .master("local[*]")
                .appName("SparkFPGrowth")
                .getOrCreate();
        List<Row> data = Arrays.asList(
                RowFactory.create(Arrays.asList("a b c".split(" "))),
                RowFactory.create(Arrays.asList("a b c d".split(" "))),
                RowFactory.create(Arrays.asList("a b".split(" ")))
        );
        StructType schema = new StructType(new StructField[]{new StructField(
                "items", new ArrayType(DataTypes.StringType, true), false, Metadata.empty())
        });
        Dataset<Row> itemsDF = spark.createDataFrame(data, schema);
        FPGrowthModel model = new FPGrowth().setItemsCol("items").setMinSupport(0.3).setMinConfidence(0.5).fit(itemsDF);
        List<String> IF = new ArrayList<>();
        for (char i = 'a'; i < 'c'; i++){
            IF.add(String.valueOf(i));
        }
        Map<String,Double> recommendList = algoRecommend1(model,IF);
        for (Map.Entry<String, Double> entry : recommendList.entrySet()){
            System.out.println(entry.getKey() + ": " + entry.getValue());
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
                    long intr = countSize;
                    double ms = ((double) (intr * intr) / (antc.size() * IF.size()));
                    double val = recommendList.get(f);
                    val += val + c * ms;
                    recommendList.put(f, val);
                }
            }
        }
        return recommendList;
    }
}
# Recommendation Algorithm 
## Preprocess Data
In this project i change data to number to echance efficiency of algorithm. After have recommned food response, i use reverse map to mapping data again and output it for user
* Create data train to number 
``` java
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
```
* Create map for mapping data used for change data train to number
```` java
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
````
* Create reverse map after have recommend list then convert it to name of food
```` java
    public static Map<Integer, String> reverseDataMapping(Map<String, Integer> data){
        Map<Integer, String> reverseData = new HashMap<>();
        for (Map.Entry<String, Integer> entry : data.entrySet()){
            reverseData.put(entry.getValue(), entry.getKey());
        }
        return reverseData;
    }
````
## Algorithm 1: Build based association rules
```` java
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
````
## Algorithm 2: Build based transactional item confidence
``` java
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
````
## Algorithm 3: Build based pair wise association rules
```` java
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
                for (Map.Entry<String,Double> entry : CD.get(inf).entrySet()){
                    if (!data.contains(entry.getKey())) {
                        P.putIfAbsent(entry.getKey(), new ArrayList<>());
                        W.putIfAbsent(entry.getKey(), new ArrayList<>());
                        Double p = CD.get(inf).get(entry.getKey()) / OD.get(inf);
                        P.get(entry.getKey()).add(p);
                        W.get(entry.getKey()).add(OD.get(inf));
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
````
## Algorithm 4: Build using Constraint Leased Recommend
``` Pesu code of algorithm 4
    function trainConstraint
    input:
    data: list data use for recommend 
    if: input food of user
    output:
        cm: car 
    OD[if] ← ø, khởi tạo danh sách food cần search
    CD[if] ← ø, danh sách các món xuất hiện cùng với các input
    for meal m ∈ data 
        if m contains all if 
        OD[if] ← OD[if] + 1 
        for each food f ∈ m & if not contains f
        if (f ∉CD[if]) 
            CD[if,f] ← 0
            CD[if, f] ← CD[if, f] + 1
        cm ← [CD, OD]
    return cm
```


```` java 
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
```` 
# Released
* 100 line of data for demo: [model-ver1](https://drive.google.com/file/d/1ZD_bd8BsI6oN5bN7pp5yQLv_tpPdow6Z/view?usp=sharing) (100 line of dataset)
* 200 line of data for demo: [model-ver2](https://drive.google.com/file/d/1_W00ewwLH3mZkuvVlZBGT3sMuE9M5zxr/view?usp=sharing) (215 line of dataset)
* Use jar file to load model: [package](https://drive.google.com/file/d/1mM_7S6Iaf6oZob3HEPNZXkIXHiQ3huOy/view?usp=sharing)
# Precision, Recall and F-measure of 4 algorithm 
![Precision và Recall](https://github.com/user-attachments/assets/8b611cfb-7bf4-4694-aa4b-bc8e705a21b7)
![F1-score](https://github.com/user-attachments/assets/3dc427d3-b2fe-42e5-a827-a818cd78b07e)
# Running Time 
![chart](https://github.com/user-attachments/assets/6574da20-e8a4-4156-88cd-e62e27ce3335)
# Summary
The CAR algorithm demonstrates a fast runtime and provides highly accurate recommendations that closely align with the algorithm's intent. Following that, the PAR and TIC algorithms also show high accuracy, but they tend to generate many redundant rules during the rule generation process. This leads to less efficient and stable performance compared to CAR.
# Contact
````If you have any contributions, please fork this repository and make a pull request ````
Feel free to submit your contributions 💌
# Preferences
     Timur Osadchiy, Ivan Poliakov, Patrick Olivier, Maisie Rowland, Emma Foster,
     Corrigendum to “Recommender system based on pairwise association rules” [Expert Systems With Applications 115 (2018) 535–542],
     Expert Systems with Applications,
     Volume 135,
     2019,
     Page 410,
     ISSN 0957-4174,
     https://doi.org/10.1016/j.eswa.2019.05.022.
     (https://www.sciencedirect.com/science/article/pii/S0957417419303513)

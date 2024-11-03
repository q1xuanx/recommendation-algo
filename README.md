# Recomedation Algorithm 
## Algorithm 1: Build based association rules
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
## Algorithm 2: Build based transactional item confidence
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
## Algorithm 3: Build based pair wise association rules
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
# I have releases first version of project: 
* 100 line of data for demo: [model-ver1](https://drive.google.com/file/d/1ZD_bd8BsI6oN5bN7pp5yQLv_tpPdow6Z/view?usp=sharing) (100 line of data)
* Use jar file to load model: [package](https://drive.google.com/file/d/1mM_7S6Iaf6oZob3HEPNZXkIXHiQ3huOy/view?usp=sharing)
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

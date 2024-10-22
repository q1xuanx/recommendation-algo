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

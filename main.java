import com.sun.org.apache.xalan.internal.xsltc.util.IntegerArray;
import com.sun.xml.internal.ws.policy.privateutil.PolicyUtils;
import org.apache.commons.lang.ArrayUtils;
import org.apache.commons.math.util.DoubleArray;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.MinHashLSH;
import org.apache.spark.ml.feature.MinHashLSHModel;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.mllib.clustering.PowerIterationClustering;
import org.apache.spark.mllib.clustering.PowerIterationClusteringModel;
import org.apache.spark.rdd.PairRDDFunctions$;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.codehaus.janino.Java;
import scala.Tuple2;
import scala.Tuple3;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.*;

import static org.apache.spark.sql.functions.col;

public class Shingling {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setMaster("local").setAppName("Shingling");
        JavaSparkContext context = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().master("local").appName("Shingling").getOrCreate();

        JavaRDD<String> input = context.textFile("data/Data.txt");
        System.out.print(input.toString());

        JavaPairRDD<String, Integer> map1 = input.flatMapToPair(new PairFlatMapFunction<String, String, Integer>() {

            @Override
            public Iterator<Tuple2<String, Integer>> call(String s) throws Exception {

                Collection<Tuple2<String, Integer>> allPair = new ArrayList<>();             //Creating Tuple of all Pairs
                if (s.contains("\t") && s.length() != 0 && s.charAt(0) != '\t' && s.charAt(0) != ' ') {
                    String temp[] = s.split("\t");
                    Set<String> keys = new HashSet<>();
                    int number = Integer.parseInt(temp[0]);
                    int i = 0;
                    char key[] = new char[5];
                    String keyString = "";

                    while (i + 5 < temp[1].length()) {
                        temp[1].getChars(i, i + 5, key, 0);

                        keyString = new String(key);
                        if (!keys.contains(keyString)) {
                            allPair.add(new Tuple2<>(keyString, number));
                            keys.add(keyString);
                        }
                        i = i + 1;
                    }
                }
                return allPair.iterator();
            }
        });

        final JavaPairRDD<String, Iterable<Integer>> reducer1 = map1.groupByKey();

        final int total_Shingles = Math.toIntExact(reducer1.count());

        JavaPairRDD<Tuple2<String, Iterable<Integer>>, Long> shingleId = reducer1.zipWithUniqueId();
        //  reducer1.saveAsTextFile("out");
        //shingleId.saveAsTextFile("out1");

        JavaPairRDD<Integer, Integer> paragraph_ShingleIds = shingleId.flatMapToPair(new PairFlatMapFunction<Tuple2<Tuple2<String, Iterable<Integer>>, Long>, Integer, Integer>() {
            @Override
            public Iterator<Tuple2<Integer, Integer>> call(Tuple2<Tuple2<String, Iterable<Integer>>, Long> tuple2LongTuple2) throws Exception {
                Collection<Tuple2<Integer, Integer>> allPair = new ArrayList<>();
                Iterator<Integer> iterator = tuple2LongTuple2._1._2.iterator();
                while (iterator.hasNext()) {
                    allPair.add(new Tuple2<Integer, Integer>(iterator.next(), Math.toIntExact(tuple2LongTuple2._2)));
                }
                return allPair.iterator();
            }
        });

        JavaPairRDD<Integer, Iterable<Integer>> paragraph_Ids = paragraph_ShingleIds.groupByKey();
        paragraph_Ids.saveAsTextFile("Out_Paragraph_Shingle");

        JavaRDD<Row> rows = paragraph_Ids.map(new Function<Tuple2<Integer, Iterable<Integer>>, Row>() {
            @Override
            public Row call(Tuple2<Integer, Iterable<Integer>> integerIterableTuple2) throws Exception {
                Iterator<Integer> iterator = integerIterableTuple2._2.iterator();
                IntegerArray array = new IntegerArray();

                int count = 0;
                while (iterator.hasNext()) {
                    array.add(iterator.next());
                    count++;
                }

                double arrayDouble[] = new double[count];
                Arrays.fill(arrayDouble, 1.0);

                return RowFactory.create(integerIterableTuple2._1, Vectors.sparse(total_Shingles, array.toIntArray(), arrayDouble));
            }
        });

        StructType schema = new StructType(new StructField[]{
                new StructField("id", DataTypes.IntegerType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });

        Dataset<Row> dfA = spark.createDataFrame(rows, schema);

        MinHashLSH mh = new MinHashLSH()
                .setNumHashTables(20)
                .setInputCol("features")
                .setOutputCol("hashes");

        MinHashLSHModel model = mh.fit(dfA);

        //model.transform(dfA).javaRDD().saveAsTextFile("out3");

        JavaRDD<Tuple3<Long, Long, Double>> similarities = model.approxSimilarityJoin(dfA, dfA, 1.0, "JaccardDistance")
                .select(col("datasetA.id").alias("idA"),
                        col("datasetB.id").alias("idB"),
                        col("JaccardDistance")).javaRDD().map(new Function<Row, Tuple3<Long, Long, Double>>() {
                    @Override
                    public Tuple3<Long, Long, Double> call(Row row) throws Exception {
                        Integer a = row.getInt(0);
                        Integer b = row.getInt(1);
                        return new Tuple3<Long, Long, Double>(a.longValue(), b.longValue(), (1-row.getDouble(2)));//Change1
                    }
                });


        JavaPairRDD<Double, Tuple2<Long, Long>> Para_Pair = similarities.mapToPair(new PairFunction<Tuple3<Long, Long, Double>, Double, Tuple2<Long, Long>>() {
            @Override
            public Tuple2<Double, Tuple2<Long, Long>> call(Tuple3<Long, Long, Double> longLongDoubleTuple3) throws Exception {
                if(longLongDoubleTuple3._1().equals(longLongDoubleTuple3._2()) || longLongDoubleTuple3._1().compareTo(longLongDoubleTuple3._2()) == 1){
                    return new Tuple2<Double, Tuple2<Long, Long>>(0.0, new Tuple2<Long, Long>(0L,0L));//Change2
                }
                else {
                    return new Tuple2<Double, Tuple2<Long, Long>>(longLongDoubleTuple3._3(), new Tuple2<Long, Long>(longLongDoubleTuple3._1(), longLongDoubleTuple3._2()));
                }
            }
        }).sortByKey(false);

        List<Tuple2<Double, Tuple2<Long, Long>>> top_Para_Pair = Para_Pair.take(100);
        PowerIterationClustering pic = new PowerIterationClustering()
                .setK(2)
                .setMaxIterations(10);

        PowerIterationClusteringModel model2 = pic.run(similarities);

        model2.assignments().toJavaRDD().saveAsTextFile("out4");

        Para_Pair.saveAsTextFile("Out_Para_Pair");
        List<Long> book1 = new ArrayList<>();
        List<Long> book2 = new ArrayList<>();

        for (PowerIterationClustering.Assignment a : model2.assignments().toJavaRDD().collect()) {
            if(a.cluster() == 0){
                book1.add(a.id());
            }

            else{
                book2.add(a.id());
            }
        }

        int countA = 0;
        //int countB = 0;

        List<Tuple2<Long, Long>> top_Book= new ArrayList<>();
        //List<Tuple2<Long, Long>> top_Book2 = new ArrayList<>();

        final List<Long> paragraphs = new ArrayList<>();
        for(int i=0;i<100;i++){
            if(countA<5){
                top_Book.add(new Tuple2<>(top_Para_Pair.get(i)._2._1, top_Para_Pair.get(i)._2._2));
                paragraphs.add(top_Para_Pair.get(i)._2._1);
                paragraphs.add(top_Para_Pair.get(i)._2._2);
                countA++;
            }
            if(countA==5){
                break;
            }
//            else if(countB<5){
//                top_Book2.add(new Tuple2<>(top_Para_Pair.get(i)._2._1, top_Para_Pair.get(i)._2._2));
//                paragraphs.add(top_Para_Pair.get(i)._2._1);
//                paragraphs.add(top_Para_Pair.get(i)._2._2);
//                countB++;
//            }
//            if(countA==5 && countB==5){
//                break;
//            }
        }

        Collections.sort(book1);
        Collections.sort(book2);

        System.out.println("Book1= "+ String.valueOf(book1.size()) + book1);
        System.out.println("Book2= "+ String.valueOf(book2.size()) + book2);

        System.out.println("Book Top Pair:"+top_Book);
//        System.out.println("Book2 Top Pair:"+top_Book2);

        JavaPairRDD<Integer, String> map2 = input.flatMapToPair(new PairFlatMapFunction<String, Integer, String>() {
            @Override
            public Iterator<Tuple2<Integer, String>> call(String s) throws Exception {
                Collection<Tuple2<Integer, String >> allPair = new ArrayList<>();
                if (s.contains("\t") && s.length() != 0 && s.charAt(0) != '\t' && s.charAt(0) != ' ') {
                    String temp[] = s.split("\t");
                    Integer number = Integer.parseInt(temp[0]);
                    if(paragraphs.contains(number.longValue())){
                        int i = 0;
                        char key[] = new char[5];
                        String keyString = "";
                        while (i + 4 < temp[1].length()) {
                            temp[1].getChars(i, i + 5, key, 0);
                            keyString = (new String(key));
                            allPair.add(new Tuple2<Integer, String>(number, keyString));
                            i = i + 1;
                        }
                    }
                }
                return allPair.iterator();
            }
        });

        //JavaPairRDD<Integer, Iterable<String>> reduce_Shingles = map2.groupByKey();

        String bookFinal = "";
        //BOOK1

        for(int i=0;i<top_Book.size();i++){
            Long numA = top_Book.get(i)._1;
            List<String> lookup1_Shingles = map2.lookup(numA.intValue());

            Long numB = top_Book.get(i)._2;
            List<String> lookup2_Shingles = map2.lookup(numB.intValue());

            for(int j=0;j<lookup1_Shingles.size();j++){
                if(lookup2_Shingles.contains(lookup1_Shingles.get(j))){
                    int index_Shingle2 = lookup2_Shingles.indexOf(lookup1_Shingles.get(j));
                    bookFinal += "Para No1=" + numA.toString() + " - " + "Para No2=" + numB.toString() + " Shingle=" + lookup1_Shingles.get(j) + " Index1=" + String.valueOf(j) + " - " + "Index2=" + String.valueOf(index_Shingle2) + "\n";
                }
            }

            bookFinal += "-------------------------------------------\n";

        }

        System.out.println(bookFinal);


        //BOOK2
//        String book2Final = "";
//        for(int i=0;i<top_Book2.size();i++){
//            Long numA = top_Book2.get(i)._1;
//            List<String> lookup1_Shingles = map2.lookup(numA.intValue());
//
//            Long numB = top_Book2.get(i)._2;
//            List<String> lookup2_Shingles = map2.lookup(numB.intValue());
//
//            for(int j=0;j<lookup1_Shingles.size();j++){
//                if(lookup2_Shingles.contains(lookup1_Shingles.get(j))){
//                    int index_Shingle2 = lookup2_Shingles.indexOf(lookup1_Shingles.get(j));
//                    book2Final += "Para No.=" + numA.toString() + " - " + "Para No=" + numB.toString() + " Shingle=" + lookup1_Shingles.get(j) + " Index1=" + String.valueOf(j) + " - " + "Index2="+ String.valueOf(index_Shingle2) + "\n";
//                }
//            }
//            book2Final += "-------------------------------------------\n";

//        }

        //System.out.println(book2Final);

        try {
            try (PrintStream out = new PrintStream(new FileOutputStream("Out_Top5Para.txt"))) {
                out.print("---------------------------------\n");
                out.print("Book1\n");
                out.print("----------------------------------\n");
                out.print(book1);

                out.print("\n----------------------------------\n");
                out.print("Book2\n");
                out.print("----------------------------------\n");
                out.print(book2);
                out.print("\n-------------Shingles-----------------\n");

                out.print(bookFinal);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


}


import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import utils.Util._


object Grid{
  def main(args:Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\火狐下载\\hadoop-common-2.2.0-bin-master")
    val spark = SparkSession
      .builder()
      .appName("Grid")
      .master("local[6]")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    val sqlContext=spark.sqlContext
    import spark.implicits._

    val trainFile="file:///F:/share/RSSpark/uid_plays_201809_train.csv"
    val testFile="file:///F:/share/RSSpark/uid_plays_201809_test.csv"
    val trainingRDD0 =getUserItemRDD(spark,sqlContext,trainFile)

    val splits =
//      sc.textFile("file:///F:/tmp/cs.csv").map(_.split(',') match {
//      case Array(user, item, rate) =>
//        (user.toInt, item.toInt, rate.toFloat)
//    }).toDF("userId","itemId","rating")
    trainingRDD0.randomSplit(Array[Double](0.8, 0.2))

    val trainingRDD = splits(0).persist()
    val testRDD = splits(1).persist()

    val rank = 5
    val maxIter = 10
    val regParam = 0.1
    //    val alsModel: MatrixFactorizationModel = ALS.trainImplicit(trainingRDD, rank, numIterations,0.01,0.05)
    val als = new ALS()
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setUserCol("userId")
      .setItemCol("itemId")
      .setRatingCol("rating")
      .setPredictionCol("predict_rating")
//    val alsModel = als.fit(trainingRDD)
    val paramGrid = new ParamGridBuilder().
      addGrid(als.rank, Array(5,10,20)).
      addGrid(als.maxIter, Array(10,15)).
      addGrid(als.regParam, Array(0.01, 0.1, 0.05)).
      build()

    // Configure an ML pipeline, which consists of one stage
    val pipeline = new Pipeline().setStages(Array[PipelineStage](als))
    // CrossValidator 需要一个Estimator,一组Estimator ParamMaps, 和一个Evaluator.
    // （1）Pipeline作为Estimator;
    // （2）定义一个RegressionEvaluator作为Evaluator，并将评估标准设置为“rmse”均方根误差
    // （3）设置ParamMap
    // （4）设置numFolds

    val cv=new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator()
        .setLabelCol("rating")
        .setPredictionCol("predict_rating")
        .setMetricName("rmse"))
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2);

    // 运行交叉检验，自动选择最佳的参数组合
    val cvModel: CrossValidatorModel =cv.fit(trainingRDD);
    //保存模型
    cvModel.save("F:/tmp/cvModel_als.modle");
//    val sameCV = CrossValidator.load("myCVPath")
//    cvModel.save("file:///F:/tmp/cvModel_als.modle");

    //System.out.println("numFolds: "+cvModel.getNumFolds());
    //Test数据集上结果评估
    val predictions=cvModel.transform(testRDD).filter("(predict_rating between 0 and 1) and (rating between 0 and 1)")
    println(predictions.schema)
    predictions.show()
    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")     //RMS Error
      .setLabelCol("rating")
      .setPredictionCol("predict_rating");
    val rmse = evaluator.evaluate(predictions);
    System.out.println("RMSE @ test dataset " + rmse);
    //Output: RMSE @ test dataset 0.943644792277118


  }
}

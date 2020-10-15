
import java.util

import org.apache.spark.ml.feature._
import org.apache.spark.ml.fpm.{FPGrowth, FPGrowthModel}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.classification.FMClassificationModel
import org.apache.spark.ml.classification.FMClassifier
import org.apache.spark.ml.evaluation.RegressionEvaluator

object FMtest{
  List(1,2,3).reduce((a,b) => if (a > b) a else b )
  def main(args: Array[String]): Unit ={
    val spark = initSpark("test")
    val tagPath="file:///F:/tmp/ml/output/tag/20190729/7/"
    val savePathMl="file:///F:/tmp/ml/output/20190729/"
    val savePathMlB="file:///F:/tmp/ml/output/20190729/testB/"
    import org.apache.spark.mllib.util.MLUtils.saveAsLibSVMFile
    spark.udf.register("getIndexV",(v:org.apache.spark.ml.linalg.Vector,index:Int) => {v.toArray(index)} )

    val vecCols=Array("countrycode_index","itemID_index","sex_index","ua_index", "bpDay","activedays","userOnlineMusicNum","userOnlinePlayNum","userOnlinePassNum","userCollectNum","onlineplaynum","onlinepassnum","playUsernum","pop","collectNum","collectUserNum","publicDays"
      ,"songCollectRatio","userPassRatio","songPassRatio","likedMusic2vec","category2vec","userAvgDayPlay","useActiveRatio","userRepeatNum"
      ,"singerid_index","singerCollectNum","singerCollectUserNum")
    val lgbmAssembler = new VectorAssembler().setInputCols(vecCols).setOutputCol("gbdtFeature")
    //    val lgbmTrain = lgbmAssembler.transform(train_index.selectExpr("play","countrycode_index","itemID_index","sex_index","ua_index", "bpDay","activedays","userOnlineMusicNum","userOnlinePlayNum","userOnlinePassNum","userCollectNum","onlineplaynum","onlinepassnum","playUsernum","pop","collectNum","collectUserNum","publicDays"
    //      ,"songCollectRatio","userPassRatio","songPassRatio","likedMusic2vec","category2vec"
    //      ,"userAvgDayPlay","useActiveRatio","userRepeatNum","singerid_index","singerCollectNum","singerCollectUserNum"))
    //    println("save lgbmTrainDF ...")

    val lgbmTrainDF=spark.read.load(savePathMl+"lgbmSongTrainDF").selectExpr("play","gbdtFeature")
    lgbmTrainDF.show(5,false)
    println("training  ...")
    println(lgbmTrainDF.count())
    val splited = lgbmTrainDF.randomSplit(Array(0.9,0.1),2L)
    var train_ = splited(0)
    var test_ = splited(1)

//    val data=spark.read.format("com.databricks.spark.csv").option("header", "true")
//      .load("file:///F:\\python\\data\\0920m\\data\\10.txt")
//      .na.fill("-1")
//      .selectExpr("uid","itemid","country","cast(rating as double)","if(playNum>2,1,0) click","cast(disNum as double)","cast(collectNum as double)","cast(penalty as double)","genre","artistid")
//      .limit(3).cache()
//    val hasher = new FeatureHasher()
//      .setInputCols("uid","itemid","country","genre","artistid","disNum","collectNum","penalty")
////      .setInputCols("uid","itemid","country","genre","artistid","disNum","collectNum","penalty")
//      .setOutputCol("feature")
//    println("特征Hasher编码：")
//    val splited = hasher.transform(data)//.randomSplit(Array(0.9,0.1),2L)
//    var train_ = splited//(0)
//    var test_ = splited//(1)
//    train_.show(false)
//    println(train_.schema)
//    val v=new org.apache.spark.ml.linalg.SparseVector(262144,Array(159120,177308,184018,194093,197131,214751,224492,241900),Array(1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0))
//    org.apache.spark.ml.linalg.Vectors.sparse(262144,Array(159120,177308,184018,194093,197131,214751,224492,241900),Array(1.0,1.0,1.0,0.0,1.0,1.0,0.0,1.0))

//    val splited0 = data.randomSplit(Array(0.7,0.3),2L)
//    val catalog_features = Array("uid","itemid","country","click","genre","artistid")
//    var train_index = splited0(0)
//    var test_index = splited0(1)
//    for(catalog_feature <- catalog_features){
//      val indexer = new StringIndexer()
//        .setInputCol(catalog_feature)
//        .setOutputCol(catalog_feature.concat("_index"))
//      val train_index_model = indexer.fit(train_index)
//      val train_indexed = train_index_model.transform(train_index)
//      val test_indexed = indexer.fit(test_index).transform(test_index,train_index_model.extractParamMap())
//      train_index = train_indexed
//      test_index = test_indexed
//    }
//    println("字符串编码下标标签：")
//    train_index.show(false)
//    test_index.show(false)

    val classifier = new FMClassifier()
//      .setLabelCol("click")
//      .setFeaturesCol("feature")
      .setLabelCol("play")
      .setFeaturesCol("gbdtFeature")
      .setPredictionCol("predictPlay")
      .setProbabilityCol("probabilitys")
      .setMaxIter(60)
      .setFactorSize(10)
      .setStepSize(0.1)
      .setRegParam(0.02)
    val fmModel=classifier.fit(train_)

    //---------------------------3.0 模型评估：计算RMSE，均方根误差---------------------
    val predictions=fmModel.transform(test_);
    val evaluator=new RegressionEvaluator()
      .setMetricName("mse")
      .setLabelCol("play")
      .setPredictionCol("predictPlay");
    val rmse=evaluator.evaluate(predictions);
    println("test set  mse Err = " + rmse);
    predictions
//      .selectExpr("sum(play)","sum(predictPlay)","sum(if(play=1 and predictPlay=1, 1,0))","count(1)")
      .show(false)
    predictions.selectExpr("count(1)","sum(play)","sum(predictPlay)","sum(if(play=1 and predictPlay=1,1,0))").show
    auc(predictions.selectExpr("play","getIndexV(probabilitys,1) probability"),"play","probability")

    val predictionsTr=fmModel.transform(train_);
    val evaluatorTr=new RegressionEvaluator()
      .setMetricName("mse")
      .setLabelCol("play")
      .setPredictionCol("predictPlay");
    val rmseTr=evaluatorTr.evaluate(predictionsTr);
    println("train set  mse Err = " + rmseTr);
    predictionsTr.selectExpr("sum(play)","sum(predictPlay)","sum(if(play=1 and predictPlay=1, 1,0))","count(1)").show()
    auc(predictionsTr.selectExpr("play","getIndexV(probabilitys,1) probability"),"play","probability")


  }


  def saveByDate(spark:SparkSession, date:String, table:String, df:DataFrame,s:String,db:String): Unit ={
    df.repartition(6).createOrReplaceTempView("t")
    spark.sql(s"select $s FROM t").show(2)
    spark.sql(s"insert overwrite table $db.$table partition(pdate='$date') select $s FROM t")
  }

  def initSpark(appName:String,mongoCollection:String="boomplay_follow",mongoCollect:String=".follow"):SparkSession ={
    val props=System.getProperties(); //获得系统属性集
    val osName = props.getProperty("os.name"); //操作系统名称     spark.conf.set("spark.driver.maxResultSize","4G")
    if (osName.contains("Windows"))
      System.setProperty("hadoop.home.dir", "D:\\火狐下载\\hadoop-common-2.2.0-bin-master")
    val spark = if (osName.contains("Windows")) SparkSession
      .builder()
      .config("spark.sql.warehouse.dir", "file:///F://svn//sparkProject//recommend//spark-warehouse")
      .enableHiveSupport()
      .appName(appName)
      .master("local[6]")
      .config("spark.driver.maxResultSize","4G")
      .config("spark.driver.memory","30G")
      .config("spark.cores.max","6")
      .config("spark.redis.host", "localhost")
      .config("spark.redis.port", "6379")
      .config("spark.mongodb.input.uri", "mongodb://10.200.60.16:20016/boomplay_follow.follow")
      .getOrCreate()
    else  SparkSession
      .builder()
      .enableHiveSupport()
      .appName(appName)
//      .config("spark.mongodb.input.uri", Conf.mongo+mongoCollection+mongoCollect)
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    // 设置运行环境 spark.default.parallelism
    spark.conf.set("spark.kryoserializer.buffer.mb", "8")
    spark
  }

  def auc (df:DataFrame,labelCol:String,predictCol:String){
    var count = 0
    val dfLen=df.count()
    if(dfLen<50000) {
      val p = df.filter(s"$labelCol >=1").select(s"$predictCol").rdd.map(r => (r.getDouble(0))).collect()
      val n = df.filter(s"$labelCol <1").select(s"$predictCol").rdd.map(r => (r.getDouble(0))).collect()
      for (rowp <- p) {
        for (rown <- n) {
          if (rowp > rown)
            count = count + 1
        }
      }
      val aucValue= count*1.0 / ( p.length * n.length)
      println( s"AUC = $count/(${p.length}*${n.length}) 即：AUC = $aucValue")
    }else{ // 太多就抽取一万条
      val samp=10000.0/dfLen
      val df2=df.sample(samp).persist()
      val p = df2.filter(s"$labelCol >=1").select(s"$predictCol").rdd.map(r => (r.getDouble(0))).collect()
      val n = df2.filter(s"$labelCol <1").select(s"$predictCol").rdd.map(r => (r.getDouble(0))).collect()
      for (rowp <- p) {
        for (rown <- n) {
          if (rowp > rown)
            count = count + 1
        }
      }
      val aucValue= count*1.0 / ( p.length * n.length)
      println( s"AUC = $count/(${p.length}*${n.length}) 即：AUC = $aucValue")    }

  }


}
import org.apache.spark.ml.recommendation.ALSModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.encoders.RowEncoder
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

import scala.collection.{immutable, mutable}
import org.mariadb.jdbc

object test{
  def main(args: Array[String]): Unit ={
    val spark = initSpark("Normalize",Conf.sparkMaster,Conf.warehouse)
    
  }

  def lr(spark:SparkSession): Unit = {
    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.feature.FeatureHasher
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    val data = spark.read.csv("file:///F:/R/iris0.csv")
      .toDF("sl","sw","pl","pw","t")
        .selectExpr("sl","sw","pl","pw","cast(t as Int)")
    data.show(5,false)
    val splited = data.randomSplit(Array(0.8,0.2),2L)
    var trainDF = splited(0)
    var testDF = splited(1)
    val hasher = new FeatureHasher()
      .setInputCols("sl","sw","pl","pw")
      .setOutputCol("feature")
    println("特征Hasher编码：")
    val train_hs = hasher.transform(trainDF)
    val test_hs = hasher.transform(testDF)
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0)
      .setFeaturesCol("feature")
      .setLabelCol("t")
      .setPredictionCol("click_predict")
    val model_lr = lr.fit(train_hs)
    println(s"每个特征对应系数: ${model_lr.coefficients} 截距: ${model_lr.intercept}")
    val predictions = model_lr.transform(test_hs)
    predictions.select("t","click_predict","probability").show(100,false)
    val predictionRdd = predictions.select("click_predict","t").rdd.map{
      case Row(click_predict:Double,click_index:Double)=>(click_predict,click_index.toDouble)
    }
    val metrics = new MulticlassMetrics(predictionRdd)
    val accuracy = metrics.accuracy
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure
    println(s"LR评估结果：\n分类正确率：${accuracy}\n加权正确率：${weightedPrecision}\n加权召回率：${weightedRecall}\nF1值：${f1}")
  }
  def lr0(spark:SparkSession): Unit = {
    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.feature.FeatureHasher
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    val data = spark.read.csv("/opt/data/ads_6M.csv").toDF(
      "id","click","hour","C1","banner_pos","site_id","site_domain",
      "site_category","app_id","app_domain","app_category","device_id","device_ip",
      "device_model","device_type","device_conn_type","C14","C15","C16","C17","C18",
      "C19","C20","C21")
    data.show(5,false)
    val splited = data.randomSplit(Array(0.7,0.3),2L)
    val catalog_features = Array("click","site_id","site_domain","site_category","app_id","app_domain","app_category","device_id","device_ip","device_model")
    var train_index = splited(0)
    var test_index = splited(1)
    for(catalog_feature <- catalog_features){
      val indexer = new StringIndexer()
        .setInputCol(catalog_feature)
        .setOutputCol(catalog_feature.concat("_index"))
      val train_index_model = indexer.fit(train_index)
      val train_indexed = train_index_model.transform(train_index)
      val test_indexed = indexer.fit(test_index).transform(test_index,train_index_model.extractParamMap())
      train_index = train_indexed
      test_index = test_indexed
    }
    println("字符串编码下标标签：")
    train_index.show(5,false)
    test_index.show(5,false)
    //    特征Hasher
    val hasher = new FeatureHasher()
      .setInputCols("site_id_index","site_domain_index","site_category_index","app_id_index","app_domain_index","app_category_index","device_id_index","device_ip_index","device_model_index","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21")
      .setOutputCol("feature")
    println("特征Hasher编码：")
    val train_hs = hasher.transform(train_index)
    val test_hs = hasher.transform(test_index)
    /**
      * LR建模
      * setMaxIter设置最大迭代次数(默认100),具体迭代次数可能在不足最大迭代次数停止(见下一条)
      * setTol设置容错(默认1e-6),每次迭代会计算一个误差,误差值随着迭代次数增加而减小,当误差小于设置容错,则停止迭代
      * setRegParam设置正则化项系数(默认0),正则化主要用于防止过拟合现象,如果数据集较小,特征维数又多,易出现过拟合,考虑增大正则化系数
      * setElasticNetParam正则化范式比(默认0),正则化有两种方式:L1(Lasso)和L2(Ridge),L1用于特征的稀疏化,L2用于防止过拟合
      * setLabelCol设置标签列
      * setFeaturesCol设置特征列
      * setPredictionCol设置预测列
      * setThreshold设置二分类阈值
      */
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0)
      .setFeaturesCol("feature")
      .setLabelCol("click_index")
      .setPredictionCol("click_predict")
    val model_lr = lr.fit(train_hs)
    println(s"每个特征对应系数: ${model_lr.coefficients} 截距: ${model_lr.intercept}")
    val predictions = model_lr.transform(test_hs)
    predictions.select("click_index","click_predict","probability").show(100,false)
    val predictionRdd = predictions.select("click_predict","click_index").rdd.map{
      case Row(click_predict:Double,click_index:Double)=>(click_predict,click_index)
    }
    val metrics = new MulticlassMetrics(predictionRdd)
    val accuracy = metrics.accuracy
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure
    println(s"LR评估结果：\n分类正确率：${accuracy}\n加权正确率：${weightedPrecision}\n加权召回率：${weightedRecall}\nF1值：${f1}")
  }
  def saveByDate(spark:SparkSession, date:String, table:String, df:DataFrame,s:String,db:String): Unit ={
    df.repartition(8).createOrReplaceTempView("t")
    spark.sql(s"insert overwrite table $db.$table partition(pdate='$date') select $s FROM t")
  }


    def gbdtLr(spark:SparkSession): Unit = {
    import org.apache.spark.ml.feature.StringIndexer
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.feature.FeatureHasher
    import org.apache.spark.mllib.evaluation.MulticlassMetrics
    val data0=spark.read.format("com.databricks.spark.csv").option("header", "true")
      .load("file:///F:\\python\\data\\0920m\\data\\10.txt")
    val data=data0
//      spark.read.format("com.databricks.spark.csv").option("header", "true")
//      .load("file:///F:\\python\\data\\0920m\\data\\part-00000-ca88c467-c550-43ec-9341-090c14a0dfca-c000.csv").limit(10)
      .na.fill("-1")
      .selectExpr("uid","itemid","country","cast(rating as double)","if(playNum>2,1,0) click","cast(disNum as double)","cast(collectNum as double)","cast(penalty as double)","genre","artistid")

    data.show(false)
    val splited = data.randomSplit(Array(0.7,0.3),2L)
    val catalog_features = Array("uid","itemid","country","click","genre","artistid")
    var train_index = splited(0)
    var test_index = splited(1)
    for(catalog_feature <- catalog_features){
      val indexer = new StringIndexer()
        .setInputCol(catalog_feature)
        .setOutputCol(catalog_feature.concat("_index"))
      val train_index_model = indexer.fit(train_index)
      val train_indexed = train_index_model.transform(train_index)
      val test_indexed = indexer.fit(test_index).transform(test_index,train_index_model.extractParamMap())
      train_index = train_indexed
      test_index = test_indexed
    }
    println("字符串编码下标标签：")
    train_index.show(false)
    test_index.show(false)

    println("类似VectorAssembler...")
    //    特征转组合feature，类似VectorAssembler
    val hasher = new FeatureHasher()
      .setInputCols("uid_index","itemid_index","country_index","genre_index","artistid_index","rating","disNum","collectNum","penalty")
      .setOutputCol("feature")
    println("特征Hasher编码：")
    val train_hs = hasher.transform(train_index)
    val test_hs = hasher.transform(test_index)

//    println("trainning gbdt...实在太慢了，还不如LGBM 单机。。。")
//    import org.apache.spark.ml.classification.GBTClassifier
//    val model = new GBTClassifier()
//      .setLabelCol("click")
//      .setFeaturesCol("feature")
//      .setPredictionCol("click_predict")
//      .fit(train_hs)
//
//    // Evaluate model on test instances and compute test error
//    val labelsAndPredictions = test_hs.rdd.mapPartitions(it=>{
//      it.map(r => {
//        val click = r.getAs[Int]("click")
//        val v = r.getAs[org.apache.spark.ml.linalg.Vector]("feature")
//        println(v)
//        val prediction = model.predict(v)
//        (click, prediction)
//      })
//    })
//    val testMSE = labelsAndPredictions.map{ case(v, p) => math.pow((v - p), 2)}.mean()
//    println(s"Test Mean Squared Error = $testMSE")
//    println(s"Learned regression GBT model:\n ${model.toDebugString}")


    /**
      * LR建模
      * setMaxIter设置最大迭代次数(默认100),具体迭代次数可能在不足最大迭代次数停止(见下一条)
      * setTol设置容错(默认1e-6),每次迭代会计算一个误差,误差值随着迭代次数增加而减小,当误差小于设置容错,则停止迭代
      * setRegParam设置正则化项系数(默认0),正则化主要用于防止过拟合现象,如果数据集较小,特征维数又多,易出现过拟合,考虑增大正则化系数
      * setElasticNetParam正则化范式比(默认0),正则化有两种方式:L1(Lasso)和L2(Ridge),L1用于特征的稀疏化,L2用于防止过拟合
      * setLabelCol设置标签列
      * setFeaturesCol设置特征列
      * setPredictionCol设置预测列
      * setThreshold设置二分类阈值
      */

    val vectorIndexerModel: VectorIndexerModel = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").fit(data)

    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0)
      .setFeaturesCol("feature")
      .setLabelCol("click")
      .setPredictionCol("click_predict")
    val model_lr = lr.fit(train_hs)
    println(s"每个特征对应系数: ${model_lr.coefficients} 截距: ${model_lr.intercept}")
    val predictions = model_lr.transform(test_hs)
    predictions.select("click_index","click_predict","probability").show(100,false)
    val predictionRdd = predictions.select("click_predict","click_index").rdd.map{
      case Row(click_predict:Double,click_index:Double)=>(click_predict,click_index)
    }
    val metrics = new MulticlassMetrics(predictionRdd)
    val accuracy = metrics.accuracy
    val weightedPrecision = metrics.weightedPrecision
    val weightedRecall = metrics.weightedRecall
    val f1 = metrics.weightedFMeasure
    println(s"LR评估结果：\n分类正确率：${accuracy}\n加权正确率：${weightedPrecision}\n加权召回率：${weightedRecall}\nF1值：${f1}")
  }

  

}


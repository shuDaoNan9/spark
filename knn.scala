
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

import scala.collection.{immutable, mutable}
//import org.apache.spark.ml.clustering.BisectingKMeans
//import org.apache.spark.ml.linalg
//import org.apache.spark.ml.linalg.{Vectors,Vector}
//import org.apache.spark.ml.clustering.{KMeans,KMeansModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.recommendation.ALSModel

object Kmeans {
  val musicPath = "file:///F:/share/music_info_2.txt"
  val savePathMl = "file:///F:/tmp/ml/"
  val savePathMllib = "file:///F:/tmp/mllib/"
  val trainFile = "file:///F:/share/train1/"
  val testFile = "file:///F:/share/RSSpark/uid_plays_201809_test.csv"
  val (rank,maxIter,regParam) = (10,10,0.05) //als10 208081 0.3421384879641967
  val alsPath=savePathMl+"recommendForTest/"+"allCountry"+"_"+rank+"_"+maxIter+"_"+regParam
  val kmPath="file:///F:/tmp/ml/km/"
  val collectPath = "file:///F:/share/music_export.csv"

  def main(args: Array[String]): Unit = {
    System.setProperty("hadoop.home.dir", "D:\\火狐下载\\hadoop-common-2.2.0-bin-master")
    val spark: SparkSession = SparkSession
      .builder()
      .config("spark.sql.warehouse.dir", "file:///F://spark project//recommend//spark-warehouse")
      .enableHiveSupport()
      .appName("ALSMain")
      .master("local[6]")
      .getOrCreate()
    // 设置运行环境
    spark.conf.set("spark.driver.memory", "50G")
    spark.conf.set("spark.cores.max", "6")
    spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    spark.conf.set("spark.kryo.registrator", "MyRegistrator")
    spark.conf.set("spark.kryo.referenceTracking", "false")
    spark.conf.set("spark.kryoserializer.buffer.mb", "8")
    spark.conf.set("spark.locality.wait", "10000")

    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    clusterByMllib(spark,kmPath)
    sc.stop()
  }

  def clusterByMllib(spark:SparkSession,kmPath:String){
    import spark.sqlContext.implicits._
    import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
    import org.apache.spark.mllib.linalg.{Vector, Vectors}

    val alsModel= ALSModel.load(alsPath)
    val songDF: DataFrame =alsModel.itemFactors
//    songDF.repartition(1).write.format("json").mode("overwrite").save(kmPath+"itemFeature")
//    return
    val songRDD =songDF.map(r=> {
//      (r.getInt(0), Vectors.dense( r.getAs[Array[Double]](1) ) )
      val f: mutable.WrappedArray[Float] =r.getAs[mutable.WrappedArray[Float]](1)
      (r.getInt(0), Vectors.dense( f.map(_.toDouble).toArray ))
    }).rdd
    songRDD.cache()
    val kmodel =new KMeans()
      .setK(100)


    /*调节分类数*/
    val trdd=songRDD.map(_._2).cache()
    val itemCosts: Seq[(Int, Double)] =Seq(10,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,500,520,540,560,580,600,650,700,800,900,1000)
      .map{
      k =>
        (k,KMeans.train(trdd,k,20).computeCost(songRDD.map(_._2)) )
    }
    println("User clustering cross-validation:")
    itemCosts.foreach { case (k, cost) => println(f"WCSS for K=$k id $cost%2.2f") }

  }
  }

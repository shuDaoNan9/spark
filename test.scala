

import org.apache.spark.{HashPartitioner, SparkConf, SparkContext, TaskContext}
import org.apache.spark.sql._
import train.ALSMain.savePathMl


object test{
  //  Array(("123","1,22,3")).flatMap(x=> x._2.split(",").map(xx=> (x._1,xx)) ).foreach(println)
  def main(args:Array[String]): Unit ={

    System.setProperty("hadoop.home.dir", "D:\\火狐下载\\hadoop-common-2.2.0-bin-master")

    val spark = SparkSession
      .builder()
      .appName("cs")
      .master("local[2]")
      //  .enableHiveSupport() //如果需要访问hive，则需要添加这一个
      .getOrCreate()
    val sc=spark.sparkContext
    sc.setLogLevel("WARN")
    val sqlContext=spark.sqlContext
    test(sc,spark)
    return

  }



  def test(sc:SparkContext, spark:SparkSession): Unit ={
    import spark.implicits._
    import org.apache.spark.ml.recommendation.ALS
    val trainDF=sc.parallelize(Seq((1,1,1),(2,1,1),(3,2,1),(1,2,2)))
      .toDF("uId","itemId","Rating")
    trainDF.show()

    val als=new ALS().setAlpha(0.01)
      .setRegParam(0.05)
      .setRank(5)
      .setItemCol("itemId")
      .setUserCol("uId")
      .setRatingCol("Rating")
      .setPredictionCol("predict_rating")
    val alsModel=als.fit(trainDF)
    val res= alsModel.recommendForAllUsers(2)
    println(res.schema)
    res.show()
    val d=alsModel.recommendForAllUsers(20)
    //    println(d.schema)
    d.write.mode("overwrite").save(savePathMl+"rec")
    d.write.mode("overwrite").format("json").save(savePathMl+"rec2")
    val dd: DataFrame =spark.sqlContext.read.format("parquet").load(savePathMl+"rec")
    dd.show()
  }

}

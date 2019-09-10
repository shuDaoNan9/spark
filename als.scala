 def trainByML(trainingDF2:DataFrame,country:String): Unit ={
    import org.apache.spark.ml.recommendation.ALSModel
    import org.apache.spark.ml.evaluation.RegressionEvaluator
    import org.apache.spark.ml.recommendation.ALS

    //    val (rank,maxIter,regParam) = (10,15,0.05) //als10 208081 0.3421384879641967
    val (rank,maxIter,regParam) = (10,10,0.05) //als10 208081 0.3421384879641967
    val als = new ALS()
      .setRank(rank)
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setUserCol("uId")
      .setItemCol("itemId")
      .setRatingCol("rating")
      .setPredictionCol("predict_rating")

    val alsModel=als.fit(trainingDF2)
    //    alsModel.write.overwrite().save(savePath+"rank"+"maxIter"+"regParam"+"")
    //    val alsModel= ALSModel.load(savePath)
    //    alsModel.itemFactors.write.save(savePath+"item")
    println(alsModel.rank)
//    val predictions0=alsModel.transform(trainingDF2)
//    //    predictions0.filter("(predict_rating is not null)").show(3)
//    val predictions=predictions0//.na.drop()
//    val d=alsModel.recommendForAllUsers(8)
//    println(d.schema)
//    d.write.save(savePath+"rec")
//    //    predictions.persist(StorageLevel.MEMORY_ONLY)
//    //    predictions.show()
//    val evaluator = new RegressionEvaluator()
//      //      .setMetricName("mae")     //mae Error
//      .setMetricName("rmse")     //RMS Error
//      .setLabelCol("rating")
//      .setPredictionCol("predict_rating");
//    val rmse = evaluator.evaluate(predictions);
//    println("mae @ test dataset " + rmse,"predictions:")
    alsModel.write.overwrite().save(savePath+country+"_"+rank+"_"+maxIter+"_"+regParam+"_"+"0")
  }

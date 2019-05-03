 /**
   * 写入redis和mysql
   */
  def save(key: String, data: Array[(Long, Long)]) = {
    val redisConnect = getRedisConnet(Config.redisHosts, Config.redisPublicPost)
    val connection = DriverManager.getConnection(Config.ilogJDBCUrl, Config.ilogUser, Config.ilogPasswd)

    val statement = connection.createStatement
    statement.executeUpdate(s"DELETE from tableRank where f_key = '$key'")

    val preStatement = connection.prepareStatement(s"insert into tableRank (f_key, f_index, f_pid, f_score) value ('$key', ?, ?, ?)")

    val byteForWrite = encodeToRedisDD(data)
    redisConnect.hset("tableRank", key, byteForWrite)

    Logger.debug(key)

    var index = 1
    data.foreach( n => {
      Logger.debug(n._1)
      Logger.debug(n._2)
      preStatement.setLong(1, index)
      preStatement.setLong(2, n._1)
      preStatement.setLong(3, n._2)
      preStatement.executeUpdate();

      index += 1
    })
    connection.close
    Logger.info(s"save key:$key to tableRank sucess")
  }
  
  /**
   * @brief 获取redis句柄
   */
  def getRedisConnet(redisHosts: scala.collection.immutable.Seq[String], redisPost:Int) = {
    var returnRC:com.redis.RedisClient = null
    for (redisHost <- redisHosts if returnRC == null) {
      try {
        // 有多个redis。不清楚哪个是master。只有master才能写。所以测试下先
        returnRC = new com.redis.RedisClient(redisHost, redisPost)
        returnRC.set("JACK_DETERMINE_WRITEABLE", "TRUE")
        returnRC.del("JACK_DETERMINE_WRITEABLE")
      } catch {
        case e:Exception => Logger.info("cannot connect redis(maybe readonly):", redisHost, redisPost)
        returnRC = null
      }
    }
    returnRC
  }

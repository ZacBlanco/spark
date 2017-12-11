/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.linalg.distributed

import org.scalatest.Ignore

import java.util.Arrays

import scala.util.Random
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, norm => brzNorm, svd => brzSvd, ReducedQR => BQR}
import breeze.numerics.abs
import org.apache.spark.mllib.linalg.distributed.fastSVD
import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.util.{LocalClusterSparkContext, MLlibTestSparkContext}
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.rdd.RDD

class fastSVDSuite extends SparkFunSuite with MLlibTestSparkContext {

  val m = 4
  val n = 3
  val arr = Array(0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 0.0, 2.0, 5.0, 8.0, 1.0)
  val denseData = Seq(
    Vectors.dense(0.0, 1.0, 2.0),
    Vectors.dense(3.0, 4.0, 5.0),
    Vectors.dense(6.0, 7.0, 8.0),
    Vectors.dense(9.0, 0.0, 1.0)
  )
  val sparseData = Seq(
    Vectors.sparse(3, Seq((1, 1.0), (2, 2.0))),
    Vectors.sparse(3, Seq((0, 3.0), (1, 4.0), (2, 5.0))),
    Vectors.sparse(3, Seq((0, 6.0), (1, 7.0), (2, 8.0))),
    Vectors.sparse(3, Seq((0, 9.0), (2, 1.0)))
  )

  val principalComponents = BDM(
    (0.0, 1.0, 0.0),
    (math.sqrt(2.0) / 2.0, 0.0, math.sqrt(2.0) / 2.0),
    (math.sqrt(2.0) / 2.0, 0.0, - math.sqrt(2.0) / 2.0))
  val explainedVariance = BDV(4.0 / 7.0, 3.0 / 7.0, 0.0)

  var denseMat: RowMatrix = _
  var sparseMat: RowMatrix = _

  override def beforeAll() {
    super.beforeAll()
    denseMat = new RowMatrix(sc.parallelize(denseData, 2))
    sparseMat = new RowMatrix(sc.parallelize(sparseData, 2))
  }

  def randomMat(trows: Int, tcols: Int): RowMatrix = {
    var testMat: BDM[Double] = BDM.rand(trows, tcols)
    var x: Matrix = Matrices.dense(testMat.rows, testMat.cols, testMat.data)
    val cols = x.toArray.grouped(x.numRows)
    val rows = cols.toSeq.transpose
    val vecs = rows.map(row => new DenseVector(row.toArray))
    val rddtemp: RDD[Vector] = sc.parallelize(vecs, 2)
    new RowMatrix(rddtemp)
  }

  def showTimeDiff(t1: Long, t2: Long, msg: String = "Time difference: "): Long = {
    val diff = t2 - t1
    print(msg + diff + "ms\n")
    diff
  }

  def min(a: Int, b: Int): Int = {
    val x = if (a < b) a else b
    x
  }


  ignore("basicSVD") {
    //1110x410
    for (matrows <- 10 to 2000 by 100) {
      for (matcols <- 10 to 2000 by 100) {

        for (numk <- 10 to min(matrows, matcols) by 25) {
          var denseRowMat: RowMatrix = randomMat(matrows, matcols)
          var t1 = System.currentTimeMillis()
          val a = fastSVD.computeSVD(denseRowMat, numk)
          var t2 = System.currentTimeMillis()
          var t3 = showTimeDiff(t1, t2,
            matrows + "x" + matcols + " " + numk + " Singular, fastSVD ")

          t1 = System.currentTimeMillis()
          var v: SingularValueDecomposition[RowMatrix, Matrix] = denseRowMat.computeSVD(numk, true)
          t2 = System.currentTimeMillis()
          var t4 = showTimeDiff(t1, t2,
            matrows + "x" + matcols + " " + numk + " Singular, Spark ")
          showTimeDiff(t3, t4, "Spark - fastSVD time ")
        }
      }
    }
  }

  def matnorm(a: BDM[Double]): Double = {
    val b: BDM[Double] = a *:* a
    val c: Double = breeze.linalg.sum(b)
    math.sqrt(c)
  }
  def testSVDDiff(svd1: SingularValueDecomposition[Matrix, Matrix],
                  svd2: SingularValueDecomposition[RowMatrix, Matrix],
                  orig: BDM[Double]): (Double, Double, Double) = {
    var sdiff = svd1.s.asBreeze - svd2.s.asBreeze
    var st: Double = breeze.linalg.sum(sdiff *:* sdiff)
    val snorm: Double = math.sqrt(st)

    var u1 = svd1.U.asBreeze.asInstanceOf[BDM[Double]]
    var u2 = svd2.U.toBreeze()
    var udiff: BDM[Double] = u1 - u2
    var unorm = matnorm(udiff)

    var v1 = svd1.V.asBreeze.asInstanceOf[BDM[Double]]
    var v2 = svd2.V.asBreeze.asInstanceOf[BDM[Double]].t
    var vdiff: BDM[Double] = v1 - v2
    var vnorm = matnorm(vdiff)

    var tm = u1 * breeze.linalg.diag(svd1.s.asBreeze.asInstanceOf[BDV[Double]]) * v1
    var normdiff = matnorm(tm - orig)

    println(unorm)
//    println(snorm)
//    println(vnorm)
//    println(normdiff)
//
    println("U1")
    println(u1)
    println("U2")
    println(u2)

    println("V1")
    println(v1)
    println("V2")
    println(v2)
    if (unorm > 1) {

    }
    if (vnorm > 1) {

    }

//    assert(unorm < svd2.s.asBreeze.asInstanceOf[BDV[Double]](0)*.1)
//    assert(snorm < .1)
//    assert(vnorm < .1)
    (unorm, snorm, vnorm)
//    (0, 0, 0)
  }

  test("LU Factorization") {
//    var drm = randomMat(3, 3).toBreeze().asInstanceOf[BDM[Double]]
    val r: Array[Double] = Array(1, 3, 2, 2, 8, 6, 4, 14, 13)
    var drm: BDM[Double] = new BDM[Double](3, 3, r)
    var (pl, u) = fastSVD.LUFactorization(drm, pl_only = false)
    assert(matnorm(pl*u - drm) < 0.1)
//    for (x <- 4 to 100) {
//      drm = randomMat(x, x).toBreeze()
//      var (pl, u) = fastSVD.LUFactorization(drm, pl_only = false)
//      assert(matnorm(pl*u - drm) < 0.1)
//    }

  }
  ignore("fastSVD correctness") {
    var m = 40
    var n = 20
    var k = 10
    var drm: RowMatrix = randomMat(m, n)
    var a = fastSVD.computeSVD(drm, k, p_iter = 2)
    var b = drm.computeSVD(k, computeU = true)
    System.out.println("m: " + m + " n: " + n + " k: " + 3)
    testSVDDiff(a, b, drm.toBreeze().asInstanceOf[BDM[Double]])


//    for (m <- 10 to 100 by 25) {
//      for (n <- 10 to 100 by 25) {
//        for (k <- 5 to 75 by 20) {
//          var h: Int = min(k, min(m, n))
//          var drm: RowMatrix = randomMat(m, n)
//          var a = fastSVD.computeSVD(drm, h)
//          var b = drm.computeSVD(h, computeU = true)
//          System.out.println("m: " + m + " n: " + n + " k: " + h)
//          testSVDDiff(a, b, drm.toBreeze().asInstanceOf[BDM[Double]])
//        }
//      }
//    }


  }
}

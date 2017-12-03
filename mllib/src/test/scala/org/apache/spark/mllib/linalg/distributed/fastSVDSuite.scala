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

import java.util.Arrays

import scala.util.Random

import breeze.linalg.{norm => brzNorm, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs

import org.apache.spark.SparkFunSuite
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.random.RandomRDDs
import org.apache.spark.mllib.util.{LocalClusterSparkContext, MLlibTestSparkContext}
import org.apache.spark.mllib.util.TestingUtils._

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

  test("toBreeze") {
    val expected = BDM(
      (0.0, 1.0, 2.0),
      (3.0, 4.0, 5.0),
      (6.0, 7.0, 8.0),
      (9.0, 0.0, 1.0))
    for (mat <- Seq(denseMat, sparseMat)) {
      assert(mat.toBreeze() === expected)
    }
  }

  test("multiply a local matrix") {
    val B = Matrices.dense(n, 2, Array(0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
    for (mat <- Seq(denseMat, sparseMat)) {
      val AB = mat.multiply(B)
      assert(AB.numRows() === m)
      assert(AB.numCols() === 2)
      assert(AB.rows.collect().toSeq === Seq(
        Vectors.dense(5.0, 14.0),
        Vectors.dense(14.0, 50.0),
        Vectors.dense(23.0, 86.0),
        Vectors.dense(2.0, 32.0)
      ))
    }
  }
//
//  test("QR Decomposition") {
//    for (mat <- Seq(denseMat, sparseMat)) {
//      val result = mat.tallSkinnyQR(true)
//      val expected = breeze.linalg.qr.reduced(mat.toBreeze())
//      val calcQ = result.Q
//      val calcR = result.R
//      assert(closeToZero(abs(expected.q) - abs(calcQ.toBreeze())))
//      assert(closeToZero(abs(expected.r) - abs(calcR.asBreeze.asInstanceOf[BDM[Double]])))
//      assert(closeToZero(calcQ.multiply(calcR).toBreeze - mat.toBreeze()))
//      // Decomposition without computing Q
//      val rOnly = mat.tallSkinnyQR(computeQ = false)
//      assert(rOnly.Q == null)
//      assert(closeToZero(abs(expected.r) - abs(rOnly.R.asBreeze.asInstanceOf[BDM[Double]])))
//    }
//  }

//  test("compute covariance") {
//    for (mat <- Seq(denseMat, sparseMat)) {
//      val result = mat.computeCovariance()
//      val expected = breeze.linalg.cov(mat.toBreeze())
//      assert(closeToZero(abs(expected) - abs(result.asBreeze.asInstanceOf[BDM[Double]])))
//    }
//  }
//
//
//  test("QR decomposition should aware of empty partition (SPARK-16369)") {
//    val mat: RowMatrix = new RowMatrix(sc.parallelize(denseData, 1))
//    val qrResult = mat.tallSkinnyQR(true)
//
//    val matWithEmptyPartition = new RowMatrix(sc.parallelize(denseData, 8))
//    val qrResult2 = matWithEmptyPartition.tallSkinnyQR(true)
//
//    assert(qrResult.Q.numCols() === qrResult2.Q.numCols(), "Q matrix ncol not match")
//    assert(qrResult.Q.numRows() === qrResult2.Q.numRows(), "Q matrix nrow not match")
//    qrResult.Q.rows.collect().zip(qrResult2.Q.rows.collect())
//      .foreach(x => assert(x._1 ~== x._2 relTol 1E-8, "Q matrix not match"))
//
//    qrResult.R.toArray.zip(qrResult2.R.toArray)
//      .foreach(x => assert(x._1 ~== x._2 relTol 1E-8, "R matrix not match"))
//  }
}

//class RowMatrixClusterSuite extends SparkFunSuite with LocalClusterSparkContext {
//
//  var mat: RowMatrix = _
//
//  override def beforeAll() {
//    super.beforeAll()
//    val m = 4
//    val n = 200000
//    val rows = sc.parallelize(0 until m, 2).mapPartitionsWithIndex { (idx, iter) =>
//      val random = new Random(idx)
//      iter.map(i => Vectors.dense(Array.fill(n)(random.nextDouble())))
//    }
//    mat = new RowMatrix(rows)
//  }
//}

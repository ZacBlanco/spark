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

package org.apache.spark.mllib.linalg.distributed;

import java.util.Arrays

import breeze.linalg.svd.SVD
import org.apache.spark.annotation.Since
import org.apache.spark.mllib.linalg.{Matrix => SparkMatrix}
import org.apache.spark.mllib.linalg.Matrices
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{Matrix, MatrixSingularException, inv, DenseMatrix => BDM, DenseVector => BDV, LU => BLU, SparseVector => BSV, axpy => brzAxpy, qr => BQR, svd => brzSvd}
import breeze.numerics.{sqrt => brzSqrt}
import org.apache.spark.mllib.linalg
import org.apache.spark.rdd.RDD

object fastSVD {

  /**
   * Computes singular value decomposition of this matrix. Denote this matrix by A (m x n). This
   * will compute matrices U, S, V such that A ~= U * S * V', where S contains the leading k
   * singular values, U and V contain the corresponding singular vectors.
   *
   * At most k largest non-zero singular values and associated vectors are returned. If there are k
   * such values, then the dimensions of the return will be:
   *  - U is a RowMatrix of size m x k that satisfies U' * U = eye(k),
   *  - s is a Vector of size k, holding the singular values in descending order,
   *  - V is a Matrix of size n x k that satisfies V' * V = eye(k).
   *
   * We assume n is smaller than m, though this is not strictly required.
   * The singular values and the right singular vectors are derived
   * from the eigenvalues and the eigenvectors of the Gramian matrix A' * A. U, the matrix
   * storing the right singular vectors, is computed via matrix multiplication as
   * U = A * (V * S^-1^), if requested by user. The actual method to use is determined
   * automatically based on the cost:
   *  - If n is small (n &lt; 100) or k is large compared with n (k &gt; n / 2), we compute
   *    the Gramian matrix first and then compute its top eigenvalues and eigenvectors locally
   *    on the driver. This requires a single pass with O(n^2^) storage on each executor and
   *    on the driver, and O(n^2^ k) time on the driver.
   *  - Otherwise, we compute (A' * A) * v in a distributive way and send it to ARPACK's DSAUPD to
   *    compute (A' * A)'s top eigenvalues and eigenvectors on the driver node. This requires O(k)
   *    passes, O(n) storage on each executor, and O(n k) storage on the driver.
   *
   * Several internal parameters are set to default values. The reciprocal condition number rCond
   * is set to 1e-9. All singular values smaller than rCond * sigma(0) are treated as zeros, where
   * sigma(0) is the largest singular value. The maximum number of Arnoldi update iterations for
   * ARPACK is set to 300 or k * 3, whichever is larger. The numerical tolerance for ARPACK's
   * eigen-decomposition is set to 1e-10.
   *
   * @param a RowMatrix to perform SVD
   * @param k number of leading singular values to keep (0 &lt; k &lt;= n).
   *          It might return less than k if
   *          there are numerically zero singular values or there are not enough Ritz values
   *          converged before the maximum number of Arnoldi update iterations is reached (in case
   *          that matrix A is ill-conditioned).
   *              are treated as zero, where sigma(0) is the largest singular value.
   * @param mode: Mode to perform ARPACK operations with
                  (default = auto, otherwise, "local-svd", "local-eigs", "dist-eigs")
   * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
   *
   * @note The conditions that decide which method to use internally and the default parameters are
   * subject to change.
   */

  def lu(a: BDM[Double]): (BDM[Double], Array[Int]) = {
    BLU(a)
  }
  def transpose(a: BDM[Double]): BDM[Double] = {
    a.t
  }
//  def MatrixToRM(m: SparkMatrix): RowMatrix = {
//    val cols = m.toArray.grouped(m.numRows)
//    val rows = cols.toSeq.transpose
//    val vecs = rows.map(r => new DenseVector(r.toArray))
//    RowMatrix(sc.parallelize(vecs))
//  }

  @Since("1.0.0")
  def computeSVD(
      a: RowMatrix,
      k: Int
      ): SingularValueDecomposition[SparkMatrix, SparkMatrix] = {

    computeSVD(a, k)
  }

  def min(a: Long, b: Long): Long = if (a < b)  a else b

  /**
  * The actual SVD implementation, visible for testing.
  *
  * @param a The matrix to compute SVD on
  * @param k number of leading singular values to keep (0 &lt; k &lt;= n)
  * @param center Whether or not the data must be centered
                  (if center = true rows will be mean-centered)
  * @param p_iter The number of power iterations to conduct. defaults to 2
  * @param blk_size The block size of the normalized power iterations, defaults to k+2
  * @param mode computation mode (auto: determine automatically which mode to use,
  *             local-svd: compute gram matrix and computes its full SVD locally,
  *             local-eigs: compute gram matrix and computes its top eigenvalues locally,
  *             dist-eigs: compute the top eigenvalues of the gram matrix distributively)
  * @return SingularValueDecomposition(U, s, V)
  */
  def computeSVD(
      a: RowMatrix,
      k: Int,
      center: Boolean = false,
      p_iter: Int = 2,
      blk_size: Int = -1,
      mode: String): SingularValueDecomposition[SparkMatrix, SparkMatrix] = {
    var n = a.numCols().toInt
    require(k > 0 && k <= n, s"Requested k singular values but got k=$k and numCols=$n.")

    object SVDMode extends Enumeration {
      val LocalARPACK, LocalLAPACK, DistARPACK = Value
    }

    val computeMode = mode match {
      case "auto" =>
        if (k > 5000) {
          //logWarning(s"computing svd with k=$k and n=$n, please check necessity")
        }

        // TODO: The conditions below are not fully tested.
        if (n < 100 || (k > n / 2 && n <= 15000)) {
          // If n is small or k is large compared with n, we better compute the Gramian matrix first
          // and then compute its eigenvalues locally, instead of making multiple passes.
          if (k < n / 3) {
            SVDMode.LocalARPACK
          } else {
            SVDMode.LocalLAPACK
          }
        } else {
          // If k is small compared with n, we use ARPACK with distributed multiplication.
          SVDMode.DistARPACK
        }
      case "local-svd" => SVDMode.LocalLAPACK
      case "local-eigs" => SVDMode.LocalARPACK
      case "dist-eigs" => SVDMode.DistARPACK
      case _ => throw new IllegalArgumentException(s"Do not support mode $mode.")
    }
    var bs: Int = 0;
    if (blk_size == -1) {
      bs = k + 2
    }
    if (bs < 0) {
      throw new IllegalArgumentException("Block size for SVD must be > 0, defaults to k+2")
    }
    val m: Long = a.numRows()
    val nc = a.numCols()
    val maxK = min(m, nc)
    require(k <= maxK, "number of singular values must be less than min(rows, cols)")
    var c: BDM[Double] = null
    if (center) {
      // TODO: Center the 'a' matrix
      // *_Technically_* not finished
      // Center the matrix first to get PCA results
      c = a.toBreeze()
    }
    c = a.toBreeze()

     // Use the val "c" from here to refer to the source data matrix
     if (blk_size >= m / 1.25 || blk_size >= n / 1.25) {
       // Perform NORMAL SVD Here.
       // Return the SVD from here
       var lameSVD: SingularValueDecomposition[RowMatrix, SparkMatrix] = a.computeSVD(k, computeU = true)
       var u = lameSVD.U
       var umat = Matrices.dense(u.numRows().toInt, u.numCols().toInt, u.toBreeze().data)
       SingularValueDecomposition(umat, lameSVD.s, lameSVD.V)

     } else if (m >= n) {
       ///////////////////////////////////////////////////////
       // Step 1:
       // Generate Q matrix with values between -1 and 1.
       // Size n rows, l col
       ///////////////////////////////////////////////////////

       // Python from FBPCA
       // #
       // # Apply A to a random matrix, obtaining Q.
       // #
       // if isreal:
       //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
       // if not isreal:
       //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
       //         + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))

       // Don't worry about isreal - just do it for normal scalars

       // multiply the original data matrix with a random matrix
       // size m x n * n x blk_size ==> m x blk_size
       var q: BDM[Double] = c * ( (BDM.rand(n, blk_size) *:* 2.0) - 1.0)

       ////////////////////////////////////////////////////////
       // Step 2:
       // Perform the QR/LU decomposition
       ////////////////////////////////////////////////////////

       // Python from FBPCA
       // #
       // # Form a matrix Q whose columns constitute a
       // # well-conditioned basis for the columns of the earlier Q.
       // #
       // if n_iter == 0:
       //     (Q, _) = qr(Q, mode='economic')
       // if n_iter > 0:
       //     (Q, _) = lu(Q, permute_l=True)
       if (p_iter == 0) {
         // TODO: Come up with a way to calculate the LU factorization in a distributed fashion
         // Calculates and returns the Q from the QR factorization
         val qr: BQR.DenseQR = BQR.apply(q)
         q = qr.q
       } else if (p_iter > 0) {
         // TODO: Come up with a way to calculate the LU factorization in a distributed fashion
         // See
         // https://issues.apache.org/jira/browse/SPARK-8514
         // Calculates and returns the L from the LU factorization of the q matrix
         var (lu, _) = BLU.apply(q)
         q = lu
       }

       /////////////////////////////////////////////////////////
       // Step 3:
       // Run the power method for n_iter
       /////////////////////////////////////////////////////////

       // Python Code from FBPCA
       // #
       // # Conduct normalized power iterations.
       // #
       // for it in range(n_iter):

       //     Q = mult(Q.conj().T, A).conj().T

       //     (Q, _) = lu(Q, permute_l=True)

       //     Q = mult(A, Q)

       //     if it + 1 < n_iter:
       //         (Q, _) = lu(Q, permute_l=True)
       //     else:
       //         (Q, _) = qr(Q, mode='economic')

       for (i <- 0 to p_iter) {
         // We're not worried about conjugates - assume we're working with
         // real numbers
         // We need to write a transpose function
         q = (q.t * c).t

         // We also need a function to compute the LU factorization of Q
         var (q2: BDM[Double], _) = BLU.apply(q)
         q = c * q

         if (i + 1 < p_iter ) {
           // Compute LU
           var (q3: BDM[Double], _) = BLU.apply(q)
           q = q3
         } else {
           // Compute QR
           val qr: BQR.DenseQR = BQR.apply(q)
           q = qr.q
         }

       }


       /////////////////////////////////////////////////////////
       // Step 4:
       // SVD Q and original matrix to get singular values
       // (Assuming using BLAS?) - We should test this.
       /////////////////////////////////////////////////////////

       // # SVD Q'*A to obtain approximations to the singular values
       // # and right singular vectors of A; adjust the left singular
       // # vectors of Q'*A to approximate the left singular vectors
       // # of A.
       // #
       // QA = mult(Q.conj().T, A)
       // (R, s, Va) = svd(QA, full_matrices=False)
       // U = Q.dot(R)

       var qa = q.t * c
       // Perform local ARPACK SVD on this matrix (Or normal rowmatrix SVD?)
       val brzSvd.SVD(tempU, s, v) = brzSvd(qa)




       ////////////////////////////////////////////////////////
       // Step 5:
       // Retain only the first k rows/columns and return
       ////////////////////////////////////////////////////////

       // #
       // # Retain only the leftmost k columns of U, the uppermost
       // # k rows of Va, and the first k entries of s.
       // #
       // return U[:, :k], s[:k], Va[:k, :]

       // Truncate rows of U, s, and Va
       q = q * tempU
       var sigmas = Vectors.dense(Arrays.copyOfRange(s.data, 0, k)) // Truncated singular values
       var U = Matrices.dense(q.rows, k, q.data)
       var Va = Matrices.dense(k, v.cols, v.data)
       SingularValueDecomposition(U, sigmas, Va)

     } else if (m < n) {
//       var newA: linalg.Matrix = Matrices.fromBreeze(c.t)
//       val columns = newA.toArray.grouped(newA.numRows)
//       val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD
//       val mat = RowMatrix(rows.map(row => new DenseVector(row.toArray)))
//       computeSVD(mat, k, center, p_iter, blk_size, mode)

       var q: BDM[Double] = (( (BDM.rand(blk_size, m.toInt) :* 2.0) - 1.0) * c).t

       if (p_iter == 0) {
         val qr: BQR.DenseQR = BQR.apply(q)
         q = qr.q
       } else if (p_iter > 0) {
         var (lu, _) = BLU.apply(q)
         q = lu
       }

       for (i <- 0 to p_iter) {
         q = (c * q)
         var (q2: BDM[Double], _) = BLU.apply(q)
         q = (q.t * c).t

         if (i + 1 < p_iter ) {
           var (q3: BDM[Double], _) = BLU.apply(q)
           q = q3
         } else {
           val qr: BQR.DenseQR = BQR.apply(q)
           q = qr.q
         }
       }

       var aq = c * q
       val brzSvd.SVD(u, s, tempV) = brzSvd(aq)
       q = tempV * (q.t)

       var U = Matrices.dense(u.rows, k, u.data)
       var sigmas = Vectors.dense(Arrays.copyOfRange(s.data, 0, k))
       var Va = Matrices.dense(k, q.cols, q.data)


       SingularValueDecomposition(U, sigmas, Va)
     } else {
       null
     }
  }
}



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

import org.apache.spark.mllib.linalg.distributed.RowMatrix
import breeze.linalg.{axpy => brzAxpy, inv, svd => brzSvd, DenseMatrix => BDM, DenseVector => BDV,
  MatrixSingularException, SparseVector => BSV}
import breeze.numerics.{sqrt => brzSqrt}


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
   * @param mode: Mode to perform ARPACK operations with (default = auto, otherwise, "local-svd", "local-eigs", "dist-eigs")
   * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
   *
   * @note The conditions that decide which method to use internally and the default parameters are
   * subject to change.
   */
@Since("1.0.0")
def computeSVD(
    a: RowMatrix,
    k: Int,
    mode: String = "auto"
    ): SingularValueDecomposition[RowMatrix, Matrix] = {

  computeSVD(a, k, false, 2, -1, mode)
}

  /**
  * The actual SVD implementation, visible for testing.
  *
  * @param k number of leading singular values to keep (0 &lt; k &lt;= n)
  * @param center Whether or not the data must be centered (if center = true rows will be mean-centered)
  * @param p_iter The number of power iterations to conduct. defaults to 2
  * @param blk_size The block size of the normalized power iterations, defaults to k+2
  * @param mode computation mode (auto: determine automatically which mode to use,
  *             local-svd: compute gram matrix and computes its full SVD locally,
  *             local-eigs: compute gram matrix and computes its top eigenvalues locally,
  *             dist-eigs: compute the top eigenvalues of the gram matrix distributively)
  * @return SingularValueDecomposition(U, s, V). U = null if computeU = false.
  */
private[mllib] def computeSVD(
    a: RowMatrix,
    k: Int,
    center: Boolean = false,
    p_iter: Int = 2,
    blk_size: Int = -1, 
    mode: String): SingularValueDecomposition[RowMatrix, Matrix] = {
  val n = a.numCols().toInt
  require(k > 0 && k <= n, s"Requested k singular values but got k=$k and numCols=$n.")

  object SVDMode extends Enumeration {
    val LocalARPACK, LocalLAPACK, DistARPACK = Value
  }

  val computeMode = mode match {
    case "auto" =>
      if (k > 5000) {
        logWarning(s"computing svd with k=$k and n=$n, please check necessity")
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

  if (l == -1) {
    l = k + 2
  }
  val m = a.numRows()
  val n = a.numCols()
  val maxK = min(m, n)
  require(k <= maxK, "number of singular values must be less than min(rows, cols)")
  val c: RowMatrix
  if (center == true) {
    //TODO: Center the 'a' matrix
    c = a // NOT FINISHED
  }

  // Use the val "c" from here to refer to the source data matrix
 
  if (l >= m / 1.25 || l >= n / 1.25) {
    //Perform NORMAL SVD Here.
    // Return the SVD from here
  } else if (m >= n) {
    ///////////////////////////////////////////////////////
    // Step 1:
    // Generate Q matrix with values between -1 and 1.
    // Size n rows, 1 col
    ///////////////////////////////////////////////////////
    
    // Python from FBPCA
    //#
    // # Apply A to a random matrix, obtaining Q.
    // #
    // if isreal:
    //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l)))
    // if not isreal:
    //     Q = mult(A, np.random.uniform(low=-1.0, high=1.0, size=(n, l))
    //         + 1j * np.random.uniform(low=-1.0, high=1.0, size=(n, l)))

    ////////////////////////////////////////////////////////
    // Step 2:
    // Perform the QR decomposition
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


    ////////////////////////////////////////////////////////
    // Step 5:
    // Retain only the first k rows/columns and return
    ////////////////////////////////////////////////////////
    
    // #
    // # Retain only the leftmost k columns of U, the uppermost
    // # k rows of Va, and the first k entries of s.
    // #
    // return U[:, :k], s[:k], Va[:k, :]

  } else if (m < n) {

  }


  /////////////////////////////////////////////////
  // BELOW is the old compute SVD implementation //
  /////////////////////////////////////////////////

  // // Compute the eigen-decomposition of A' * A.
  // val (sigmaSquares: BDV[Double], u: BDM[Double]) = computeMode match {
  //   case SVDMode.LocalARPACK =>
  //     require(k < n, s"k must be smaller than n in local-eigs mode but got k=$k and n=$n.")
  //     val G = computeGramianMatrix().asBreeze.asInstanceOf[BDM[Double]]
  //     EigenValueDecomposition.symmetricEigs(v => G * v, n, k, tol, maxIter)
  //   case SVDMode.LocalLAPACK =>
  //     // breeze (v0.10) svd latent constraint, 7 * n * n + 4 * n < Int.MaxValue
  //     require(n < 17515, s"$n exceeds the breeze svd capability")
  //     val G = computeGramianMatrix().asBreeze.asInstanceOf[BDM[Double]]
  //     val brzSvd.SVD(uFull: BDM[Double], sigmaSquaresFull: BDV[Double], _) = brzSvd(G)
  //     (sigmaSquaresFull, uFull)
  //   case SVDMode.DistARPACK =>
  //     if (rows.getStorageLevel == StorageLevel.NONE) {
  //       logWarning("The input data is not directly cached, which may hurt performance if its"
  //         + " parent RDDs are also uncached.")
  //     }
  //     require(k < n, s"k must be smaller than n in dist-eigs mode but got k=$k and n=$n.")
  //     EigenValueDecomposition.symmetricEigs(multiplyGramianMatrixBy, n, k, tol, maxIter)
  // }

  // val sigmas: BDV[Double] = brzSqrt(sigmaSquares)

  // // Determine the effective rank.
  // val sigma0 = sigmas(0)
  // val threshold = rCond * sigma0
  // var i = 0
  // // sigmas might have a length smaller than k, if some Ritz values do not satisfy the convergence
  // // criterion specified by tol after max number of iterations.
  // // Thus use i < min(k, sigmas.length) instead of i < k.
  // if (sigmas.length < k) {
  //   logWarning(s"Requested $k singular values but only found ${sigmas.length} converged.")
  // }
  // while (i < math.min(k, sigmas.length) && sigmas(i) >= threshold) {
  //   i += 1
  // }
  // val sk = i

  // if (sk < k) {
  //   logWarning(s"Requested $k singular values but only found $sk nonzeros.")
  // }

  // // Warn at the end of the run as well, for increased visibility.
  // if (computeMode == SVDMode.DistARPACK && rows.getStorageLevel == StorageLevel.NONE) {
  //   logWarning("The input data was not directly cached, which may hurt performance if its"
  //     + " parent RDDs are also uncached.")
  // }

  // val s = Vectors.dense(Arrays.copyOfRange(sigmas.data, 0, sk))
  // val V = Matrices.dense(n, sk, Arrays.copyOfRange(u.data, 0, n * sk))

  // if (computeU) {
  //   // N = Vk * Sk^{-1}
  //   val N = new BDM[Double](n, sk, Arrays.copyOfRange(u.data, 0, n * sk))
  //   var i = 0
  //   var j = 0
  //   while (j < sk) {
  //     i = 0
  //     val sigma = sigmas(j)
  //     while (i < n) {
  //       N(i, j) /= sigma
  //       i += 1
  //     }
  //     j += 1
  //   }
  //   val U = this.multiply(Matrices.fromBreeze(N))
  //   SingularValueDecomposition(U, s, V)
  // } else {
  //   SingularValueDecomposition(null, s, V)
  // }
}



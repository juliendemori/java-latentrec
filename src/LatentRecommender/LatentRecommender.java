package LatentRecommender;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public class LatentRecommender {

	//max number of K
	private static int maxK = 10;

	//Defining some constants for the problem
	private static int nIter = 40;
	private static int K_param = 20;
	private static double[] lambda = new double[]{0, 0.2};
	private static int maxUser = 0;
	private static int maxMovie = 0;

	//We need to find a good value of eta
	private static double eta = 0.015;


	//Matrix containing the training errors
	private static double[][] trainErrs = new double[2][maxK];
	private static double[][] testErrs = new double[2][maxK];	

	//Defining the matrices Q and P
	private static double[][] P_matrix = null;
	private static double[][] Q_matrix = null;



	//Array of double values for the error after each iteration
	private static double[] Train_Err = new double[nIter+1];


	//The training and testing data filenames
	private static String trainingFile;
	private static String testFile;


	/*
	 * Main function
	 */
	public static void main(String[] args){
		trainingFile = args[0];
		testFile = args[1];
		try {
			DetermineDimensions(trainingFile);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		InitializeMatrices(K_param);
		RunStochasticGradientDescent(trainingFile, K_param, lambda[1]);
		P_matrix = null;
		Q_matrix = null;
		ComputeTestTrainErrors(trainingFile, testFile);
	}


	/*
	 * This is the function that will run stochastic gradient descent 40 times on every data point
	 */
	private static void RunStochasticGradientDescent(String trainingFile, int k_val, double lam) {
		try {
			Train_Err[0] = ComputeError(trainingFile, k_val, lam, false);
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		for (int iter = 0; iter < nIter; iter++){
			try {
				UpdateAll(trainingFile, k_val, lam);
			} catch (NumberFormatException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			}
			try {
				Train_Err[iter+1] = ComputeError(trainingFile, k_val, lam, false);
				System.out.println(Train_Err[iter+1]);	
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}


	/*
	 * This method computes the error of two P and Q matrices obtained
	 * by stochastic gradient descent
	 */
	private static double ComputeError(String file, int k_val, double lam, boolean test) throws IOException {
		double error = 0;
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while ((line = br.readLine()) != null){
				String[] user_movie_rating = line.split("	");
				int user = Integer.parseInt(user_movie_rating[0]);
				int movie = Integer.parseInt(user_movie_rating[1]);
				int rating = Integer.parseInt(user_movie_rating[2]);
				error = error + ComputeErrorTerm(user-1, movie-1, rating, k_val);
			}
			if (test != true){
				for (int k = 0; k < k_val; k++){
					for (int n = 0; n < maxUser; n++){
						error += lam* Math.pow(P_matrix[n][k],2);
					}
					for (int m = 0; m < maxUser; m++){
						error += lam* Math.pow(Q_matrix[m][k], 2);
					}
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return error;
	}


	/*
	 * This function computes one of the error terms in E
	 */
	private static double ComputeErrorTerm(int user, int movie, int rating, int k_val) {
		double error = rating;
		for (int k = 0; k < k_val; k++){
			error = error - (Q_matrix[movie][k]*P_matrix[user][k]);
		}
		error = Math.pow(error, 2);
		return error;
	}



	/*
	 * This function will read in all of the lines of the R matrix and update all 
	 * of the relevant entries of P and Q
	 */
	private static void UpdateAll(String file, int k_val, double lam) throws NumberFormatException, IOException {
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;

			while ((line = br.readLine()) != null) {
				String[] user_movie_rating = line.split("	");
				int user = Integer.parseInt(user_movie_rating[0]);
				int movie = Integer.parseInt(user_movie_rating[1]);
				int rating = Integer.parseInt(user_movie_rating[2]);
				UpdateOne(user-1, movie-1, rating, k_val, lam);
			}
			br.close();

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}


	/*
	 * This functionc computes the update due to one known rating
	 * It first computes the e_iu and then computes the updates to each P and Q matrix accordingly
	 */
	private static void UpdateOne(int user, int movie, int rating, int k_val, double lam) {
		double eps_iu = rating;
		for (int k = 0; k < k_val; k++){
			eps_iu = eps_iu -  P_matrix[user][k]*Q_matrix[movie][k];
		}
		eps_iu = eps_iu*2;
		//Store the current P and Q vectors
		double[] tempPvector = Arrays.copyOf(P_matrix[user], k_val);
		double[] tempQvector = Arrays.copyOf(Q_matrix[movie], k_val);

		//Now we update the P and Q matrices
		for (int k = 0; k < k_val; k++){
			P_matrix[user][k] = P_matrix[user][k] + eta*(eps_iu*tempQvector[k] - 2*lam*tempPvector[k]);
			Q_matrix[movie][k] = Q_matrix[movie][k] + eta*(eps_iu*tempPvector[k] - 2*lam*tempQvector[k]);
		}
	}


	/*
	 * This function will be called to initialize the matrices 
	 */
	private static void InitializeMatrices(int k_val) {
		double arg = ((double)5)/((double)k_val);
		double MaxInit = Math.sqrt(arg);

		//Initializing these matrices
		P_matrix = new double[maxUser][k_val];
		Q_matrix = new double[maxMovie][k_val];

		for (int k = 0; k < k_val; k++){
			for (int n = 0; n < maxUser; n++){
				P_matrix[n][k] = Math.random() * MaxInit;
			}
			for (int m = 0; m < maxMovie; m++){
				Q_matrix[m][k] = Math.random() * MaxInit;
			}
		}
	}


	/*
	 * This function will read in the file and determing the dimensions of 
	 * the P and Q matrices
	 */
	private static void DetermineDimensions(String file) throws IOException {
		try {
			BufferedReader br = new BufferedReader(new FileReader(file));
			String line;
			while ((line = br.readLine()) != null) {
				String[] user_movie = line.split("	");
				int user = Integer.parseInt(user_movie[0]);
				int movie = Integer.parseInt(user_movie[1]);
				if (user > maxUser){
					maxUser = user;
				}
				if (movie > maxMovie){
					maxMovie = movie;
				}
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}



	/*
	 * This function will run part (c) of the question 1, where we compute the test and training error for different 
	 * values of k
	 */
	private static void ComputeTestTrainErrors(String trainingFile, String testFile) {
		for (int lam = 0; lam < lambda.length; lam++){
			for (int k = 1; k < maxK+1; k++){
				InitializeMatrices(k);
				RunStochasticGradientDescent(trainingFile, k, lambda[lam]);
				try {
					trainErrs[lam][k-1] = ComputeError(trainingFile, k, lambda[lam], true);
					testErrs[lam][k-1] = ComputeError(testFile, k, lambda[lam], true);
					System.out.println(trainErrs[lam][k-1]);
					System.out.println(testErrs[lam][k-1]);
				} catch (IOException e) {
					e.printStackTrace();
				}
				P_matrix = null;
				Q_matrix = null;
			}
		}
	}
}

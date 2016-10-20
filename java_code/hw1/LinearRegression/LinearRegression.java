package LinearRegression;

import Jama.Matrix;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

/**
 * Simple Linear Regression implementation
 */
public class LinearRegression {
    public static void linearRegression() throws Exception {
        
        
//        double[][] arr_train_x = {{1,95}, {1,85}, {1,80}, {1,70}, {1,60}};
//        Matrix train_x = new Matrix(arr_train_x);
//        double[][] arr_train_y = {{85}, {95}, {70}, {65}, {70}};
//        Matrix train_y = new Matrix(arr_train_y);
//        Matrix test_x = train_x;
        
        // getMatrix(Initial row index, Final row index, Initial column index, Final column index)
//    	Matrix trainingData = MatrixData.getDataMatrix("data\\linear_regression\\linear-regression-train.csv");
//        Matrix train_x = trainingData.getMatrix(0, trainingData.getRowDimension() - 1, 0, trainingData.getColumnDimension() - 2);
//        Matrix train_y = trainingData.getMatrix(0, trainingData.getRowDimension()-1, trainingData.getColumnDimension()-1, trainingData.getColumnDimension()-1);
//        Matrix testData = MatrixData.getDataMatrix("data\\linear_regression\\linear-regression-test.csv");
//        Matrix test_x = trainingData.getMatrix(0, testData.getRowDimension() - 1, 0, testData.getColumnDimension() - 2);
//        Matrix test_y = testData.getMatrix(0, testData.getRowDimension() - 1, testData.getColumnDimension() - 1, testData.getColumnDimension() - 1);
        
        
        Matrix trainingData = MatrixData.getDataMatrix("data\\linear_regression\\linear-regression-train.csv");
        normalize(trainingData);    
        Matrix trainX = new Matrix(trainingData.getRowDimension(),  trainingData.getColumnDimension()); 
        /* add a column of 1 as bias */ 
        for (int i=0; i<trainingData.getRowDimension(); i++) {
        	trainX.set(i, 0, 1);
        }
        trainX.setMatrix(0, trainingData.getRowDimension() - 1, 1, trainingData.getColumnDimension() - 1, trainingData);
        Matrix trainY = trainingData.getMatrix(0, trainingData.getRowDimension()-1, trainingData.getColumnDimension()-1, trainingData.getColumnDimension()-1);

        Matrix testData = MatrixData.getDataMatrix("data\\linear_regression\\linear-regression-test.csv");
        normalize(testData);
        Matrix testX = new Matrix(testData.getRowDimension(), testData.getColumnDimension());
        for (int i=0; i<testData.getRowDimension(); i++) {
        	testX.set(i, 0, 1);
        }
        testX.setMatrix(0, testData.getRowDimension() - 1, 1, testData.getColumnDimension() - 1, testData);
        Matrix testY = testData.getMatrix(0, testData.getRowDimension() - 1, testData.getColumnDimension() - 1, testData.getColumnDimension() - 1);
        
        
        /* closed form */

        Matrix betaCF = getBeta(trainX, trainY);
        printMatrix(betaCF.transpose());
        System.out.println(betaCF.getRowDimension());
        Matrix predictedY = testX.times(betaCF);
        //printOutput(predictedY);
        double mseCF = getMSE(testX, testY, betaCF);
        System.out.println("Mean Square Error is: " + mseCF);
        
        /* stochastic gradient descent */
        
        Matrix betaGD = gradDescent(trainX, trainY, 1000, 0.002);
        printMatrix(betaGD.transpose());
        double mseGD = getMSE(testX, testY, betaGD);
        System.out.println("Mean Square Error is: " + mseGD);
        
        
    }
    
    private static void normalize(Matrix data){
        double[] mean = new double[data.getColumnDimension()-1];
        for (int j=0; j<data.getColumnDimension()-1; j++) {
        	for(int i=0; i<data.getRowDimension(); i++) {
        		mean[j] += data.get(i, j);
        	}
        	mean[j] /= data.getRowDimension();
        }
        
        double[] var = new double[data.getColumnDimension()-1];
        for (int j=0; j<data.getColumnDimension()-1; j++) {
        	for(int i=0; i<data.getRowDimension(); i++) {
        		var[j] += data.get(i, j)*data.get(i, j);
        	}
        	var[j] /= data.getRowDimension();
        	var[j] -= mean[j]*mean[j];
        	var[j] = Math.sqrt(var[j]);
        }
        
        for (int j=0; j<data.getColumnDimension()-1; j++) {
        	for(int i=0; i<data.getRowDimension(); i++) {
        		data.set(i,  j, (data.get(i, j)-mean[j]) / var[j] ); 
        	}
        }
    	
    }
    
    private static double getMSE(Matrix testX, Matrix testY,  Matrix beta) {
    	Matrix predictedY = testX.times(beta);
        Matrix error = predictedY.minus(testY);
        double mse = error.transpose().times(error).get(0,0) / testY.getRowDimension();
        return mse;
        
    }

    /**  @params: X and Y matrix of training data
     * returns value of beta calculated using the formula beta = (X^T*X)^ -1)*(X^T*Y)
     */
    private static Matrix getBeta(Matrix trainX, Matrix trainY) {
    
    	/****************Please Fill Missing Lines Here*****************/
        
        int num_features = trainX.getColumnDimension();
        Matrix beta = new Matrix(num_features, 1);
        beta = trainX.transpose().times(trainX).inverse().times(trainX.transpose().times(trainY));
        return beta;
    }
    
    private static Matrix gradDescent(Matrix trainX, Matrix trainY, int steps, double learningRate){
    	int num_features = trainX.getColumnDimension();
    	
    	
        Matrix beta = new Matrix(num_features, 1);
        
        /* randomly initialize beta in range (-1, 1) */ 
//        for (int i=0; i<num_features; i++) {
//        	beta.set(i, 0, Math.random()*2.0-1.0);
//        }
        

        for (int i=0; i<steps; i++){
        	double old_mse = getMSE(trainX, trainY, beta);
    
	    	Matrix sample = trainX.getMatrix(i, i, 0, trainX.getColumnDimension()-1);
	    	double predicted = sample.times(beta).get(0,0);
	    	double error = trainY.get(i, 0) - predicted;
	    	//System.out.println(predicted);
	    	
	    	
	    	beta = beta.plus(sample.transpose().times(2*learningRate*error));
	    	
//	    	System.out.print(".................................\n");
//	    	printMatrix(beta);
	    	double new_mse = getMSE(trainX, trainY, beta);
	    	System.out.println(new_mse);
	    	
//	    	if (Math.abs(new_mse - old_mse)<0.0001)
//	    		break;


        }
        
        return beta;
        
        
    }

    /**
     * @params: predicted Y matrix
     * outputs the predicted y values to the text file named "linear-regression-output"
     */
    public static void printOutput(Matrix predictedY) throws IOException {
        FileWriter fStream = new FileWriter("output\\linear_regression\\linear-regression-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for (int row =0; row<predictedY.getRowDimension(); row++) {
            out.write(String.valueOf(predictedY.get(row, 0)));
            out.newLine();
        }
        
        out.close();
    }
    
    private static void printMatrix(Matrix m) {
    	for (int i=0; i<m.getRowDimension(); i++){
    		for (int j=0; j<m.getColumnDimension(); j++) {
    			System.out.print(Math.round(m.get(i, j)*1000.0)/1000.0 + " ");
    		}
    		System.out.print("\n");
    	}
    }
}

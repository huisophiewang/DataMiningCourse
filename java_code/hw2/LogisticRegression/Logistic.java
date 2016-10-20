package LogisticRegression;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import Jama.Matrix;

/**
 * Created with IntelliJ IDEA.
 * User: tpeng
 * Date: 6/22/12
 * Time: 11:01 PM
 * To change this template use File | Settings | File Templates.
 */
public class Logistic {

    /** the learning rate */
    private double rate;

    /** the weight to learn */
    private double[] weights;
    
    private double bias; 

    /** the number of iterations */
    private int ITERATIONS = 6000;
    
    private double EPSI = Double.longBitsToDouble(971l << 52);

    public Logistic(int n) {
        this.rate = 0.001;
        weights = new double[n];
        bias = 0;
    }

    private double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    public void train(Matrix trainingData) {
    	int n = trainingData.getRowDimension();
    	int p = trainingData.getColumnDimension()-1;
        
    	//reset bias and weights
    	bias = 0;
        for (int i=0; i<weights.length; i++) {
        	weights[i] = 0.0;
        }
        
        Matrix trainX = new Matrix(n,  p+1); 
        /* add a column of 1 as bias */ 
        for (int i=0; i<n; i++) {
        	trainX.set(i, 0, 1);
        }
        trainX.setMatrix(0, n-1, 1, p, trainingData);
        //printMatrix(trainX);
        Matrix trainY = trainingData.getMatrix(0, n-1, p, p);
        //printMatrix(trainY);
        Matrix identity = new Matrix(p+1, p+1);
        for (int i=0; i<p+1; i++){
        	identity.set(i,  i, 1.0);
        }
        identity.timesEquals(0.0001);

        
        double lik = Double.NEGATIVE_INFINITY;
        double prev_lik = Double.NEGATIVE_INFINITY;
        for (int k=0; k<ITERATIONS; k++){

	        Matrix w = new Matrix(n,n);
	        Matrix predictVec = new Matrix(n, 1);
	        for (int i=0; i<n; i++) {
	        	Matrix row = trainingData.getMatrix(i, i, 0, p);
		        double[] x = row.getRowPackedCopy();
		        //System.out.println(Arrays.toString(x));
		        double predicted = classify(x);
		        predictVec.set(i, 0, predicted);
		        w.set(i, i, predicted*(1.0-predicted));
	        }
	        
	        Matrix Hessian = new Matrix(p+1, p+1);	        
	        Hessian = trainX.transpose().times(w).times(trainX);
	        // to avoid singularity	        
	        Hessian = Hessian.minus(identity);
	        //System.out.println("Hessian:");
	        //printMatrix(Hessian);
	        
	        Matrix delta = new Matrix(p+1, 1);
	        delta = trainX.transpose().times(trainY.minus(predictVec));
	        
	        Matrix update = new Matrix(p+1, 1);	        
	        update = Hessian.inverse().times(delta);

	        
	        //update weights and bias
	        /****************Please Fill Missing Lines Here*****************/
	        bias += update.get(0, 0);
	        for (int i=0; i<p;i++)  {
	            weights[i] += update.get(i+1, 0);
	        }

	        //calculate log likelihood function 
	        /****************Please Fill Missing Lines Here*****************/
	        lik = 0.0;
	        for (int i=0; i<n; i++){
	        	double z = bias;
	        	for(int j=0; j<p; j++){
	        		z += trainX.get(i, j+1)*weights[j];
	        	}
	        	lik += (trainY.get(i, 0)*z - Math.log(1+ Math.exp(z)));
	        }
	        
	        
	        System.out.println("iteration: " + k + " " + bias + " " + Arrays.toString(weights) + " mle: " + lik);
	        
	        //stop criterion
	        if (lik - prev_lik < 0.000001)
	        	break;	        
	        prev_lik = lik;
        }
        

    }

    private double classify(double[] x) {
        double logit = bias;
        for (int i=0; i<weights.length;i++)  {
            logit += weights[i] * x[i];
        }
        return sigmoid(logit);
    }
    
    public double computeAccuracy(Matrix data){
    	double acc, prob, predict;
    	int n = data.getRowDimension();
    	int m = data.getColumnDimension();
    	int num_correct = 0;
    	double[] x;
    	for (int i=0; i<n; i++){
    		Matrix row = data.getMatrix(i, i, 0, m-1);
	        x = row.getRowPackedCopy();
	        prob = classify(x);
	        if (prob >= 0.5)
	        	predict = 1.0;
        	else
        		predict = 0.0;
	        if (predict == x[m-1])
	        	num_correct += 1;
    	}
    	acc = (double)num_correct/n;
    	System.out.println(acc);
    	return acc;
    }
    
    public void printMatrix(Matrix m) {
    	for (int i=0; i<m.getRowDimension(); i++){
    		for (int j=0; j<m.getColumnDimension(); j++) {
    			System.out.print(m.get(i, j) + " ");
    		}
    		System.out.print("\n");
    	}
    }
    
    public Matrix[] partitionData(Matrix data, int fold, int k){
    	Matrix[] subsets = new Matrix[2];
    	int n = data.getRowDimension();
    	int m = data.getColumnDimension();
    	subsets[0] = new Matrix(n/fold, m);
    	subsets[1] = new Matrix(n-n/fold, m);
    	
    	int count0 = 0;
    	int count1 = 0;
    	for (int i=0; i<n; i++){
    		if (i%fold==k) {
    			for (int j=0; j<m; j++){
    				double value = data.get(i, j);
    				subsets[0].set(count0, j, value);
    			}
    			count0 += 1;
    		} else {
    			for (int j=0; j<m; j++){
    				double value = data.get(i, j);
    				subsets[1].set(count1, j, value);
    			}
    			count1 += 1;   			
    		}
    			
    	}
    	
    	return subsets;
    	
    }
    

}

package LogisticRegression;
import java.io.FileNotFoundException;
import java.io.IOException;

import Jama.Matrix;

public class Algorithms {
    public static void main(String... args) throws FileNotFoundException, IOException {
        
        Matrix data = MatrixData.getDataMatrix("data\\data.txt");
        Logistic logistic = new Logistic(data.getColumnDimension()-1);
        
	    int fold = 5;
	    double accuracy = 0.0;
	    for(int k=0; k<fold; k++) {
	    	Matrix[] subsets = logistic.partitionData(data, 5, k);
	    	logistic.train(subsets[1]);
	        accuracy += logistic.computeAccuracy(subsets[0]);
	        
	    }
	    accuracy /= fold;
	    System.out.println("Average accuracy is " + accuracy);
	    

    }
}
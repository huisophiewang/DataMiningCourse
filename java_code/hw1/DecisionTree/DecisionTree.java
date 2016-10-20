package DecisionTree;

import java.io.*;
import java.util.Arrays;
import java.util.Enumeration;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.NoSupportForMissingValuesException;
import weka.core.Utils;
import Utility.Utility;

/*Class for constructing an unpruned decision tree based on
the ID3 algorithm. Can only deal with nominal attributes.
No missing values allowed. Empty leaves may result in unclassified instances.
 */
public class DecisionTree  {
    //The node's successors.
    private DecisionTree[] m_Successors;
    //Attribute used for splitting.
    private Attribute m_Attribute;
    //Class value if node is leaf.
    private double m_ClassValue;
    //Class distribution if node is leaf.
    private double[] m_Distribution;
    // Class attribute of data set.
    private Attribute m_ClassAttribute;

    public DecisionTree() {
    }
    //Builds decision tree classifier.
    public void buildClassifier(Instances data) throws Exception {
        data = new Instances(data);
        this.makeTree(data);
    }

    private void makeTree(Instances data) throws Exception {
        if(data.numInstances() == 0) {
            this.m_Attribute = null;
            this.m_ClassValue = Instance.missingValue();
            //System.out.println("missing value: " + this.m_ClassValue);
            this.m_Distribution = new double[data.numClasses()];
        } else {
            double[] infoGains = new double[data.numAttributes()];
          
            Attribute splitData;
            for(Enumeration attEnum = data.enumerateAttributes(); attEnum.hasMoreElements(); infoGains[splitData.index()] = this.computeInfoGain(data, splitData)) {
                splitData = (Attribute)attEnum.nextElement();
            }

            System.out.println("infoGains: " + Arrays.toString(infoGains));
            
            this.m_Attribute = data.attribute(Utils.maxIndex(infoGains));
            
            System.out.println("attribute " + this.m_Attribute);
            
            if(Utils.eq(infoGains[this.m_Attribute.index()], 0.0D)) {
                this.m_Attribute = null;
                this.m_Distribution = new double[data.numClasses()];

                Instance j;
                for(Enumeration var6 = data.enumerateInstances(); var6.hasMoreElements(); ++this.m_Distribution[(int)j.classValue()]) {
                    j = (Instance)var6.nextElement();
                }

                Utils.normalize(this.m_Distribution);
                this.m_ClassValue = (double)Utils.maxIndex(this.m_Distribution);
                this.m_ClassAttribute = data.classAttribute();
                //System.out.println("Class value is: " + this.m_ClassValue);
            } else {
                Instances[] var7 = this.splitData(data, this.m_Attribute);
                this.m_Successors = new DecisionTree[this.m_Attribute.numValues()];

                for(int var8 = 0; var8 < this.m_Attribute.numValues(); ++var8) {
                    this.m_Successors[var8] = new DecisionTree();
                    this.m_Successors[var8].makeTree(var7[var8]);
                }
            }
        }
    }

    //Computes the entropy of a dataset.
    private double computeEntropy(Instances data) throws Exception {
        double[] classCounts = new double[data.numClasses()];

        Instance entropy;
        for(Enumeration instEnum = data.enumerateInstances(); instEnum.hasMoreElements(); ++classCounts[(int)entropy.classValue()]) {
            entropy = (Instance)instEnum.nextElement();
        }
        
        double totalEntropy = 0.0D;
        int classNum = data.numClasses();
        double [] classProbVec = new double[classNum];
        
        for(int j = 0; j < classNum; ++j) {
            if(classCounts[j] > 0.0D) {
                classProbVec[j]= classCounts[j]/data.numInstances();
            }
            else
            	classProbVec[j]=0;
        }
        
        System.out.println(Arrays.toString(classProbVec));

        /****************Please Fill Missing Lines Here*****************/
        for(int j = 0; j < classNum; ++j) {
        	if(classProbVec[j] > 0.0D) {
        		totalEntropy += (-classProbVec[j])*(Math.log(classProbVec[j])/Math.log(2));
        	} 
        }
        
        System.out.println(totalEntropy);

        return totalEntropy;

    }
    
    //Computes information gain for an attribute.
    private double computeInfoGain(Instances data, Attribute att) throws Exception {
        double infoGain = this.computeEntropy(data);
        System.out.println(infoGain);
        
        Instances[] splitData = this.splitData(data, att);

        /****************Please Fill Missing Lines Here*****************/
        
        double splitEntropy = 0.0D;
        for (int i=0; i<splitData.length; i++){
//        	System.out.println("========================");
//        	System.out.println(splitData[i]);
        	if (splitData[i].numInstances() > 0) {
        		double p = (double)splitData[i].numInstances() / (double)data.numInstances();
        		splitEntropy += p * this.computeEntropy(splitData[i]);
//        		System.out.println(p);
//        		System.out.println(splitEntropy);
        	}

        }
        //System.out.println(splitEntropy);
        infoGain -= splitEntropy;
        
        
        
        return infoGain;
    }

    
    
    private double computeGainRatio(Instances data, Attribute att) throws Exception {
    	double gainRatio = this.computeInfoGain(data, att);
    	//Instances[] splitData = this.splitData(data, att);
    	
    	double splitInfo = 0.0;
    	double [] attCount = new double[att.numValues()];
    	double [] attProb = new double[att.numValues()];
    	
    	for (int i=0; i<data.numInstances(); i++){
    		Instance ins = data.instance(i);
    		attCount[(int)ins.value(att)] ++;	
    	}
    			
    	for (int j=0; j<att.numValues(); j++){
    		if (attCount[j] > 0) 
    			attProb[j] = (double)attCount[j] / data.numInstances();
    		else
    			attProb[j] = 0.0;
    	}
    	
    	for (int j=0; j<att.numValues(); j++){
    		if (attProb[j]>0) {
    			splitInfo += (-attProb[j])*(Math.log(attProb[j])/Math.log(2));
    		}
    	}
    	
    	System.out.println(splitInfo);
    	// if splitInfo is 0, the data all have the same attribute value
    	// no need to consider this attribute for spliting, so set to smallest value 0
    	if (splitInfo==0) {
    		gainRatio = 0.0;
    	} else {
    		gainRatio /= splitInfo;
    	}
    	
    	
    			
    	return gainRatio;
    }
    
    //Splits a dataset according to the values of a nominal attribute.
    private Instances[] splitData(Instances data, Attribute att) {
        Instances[] splitData = new Instances[att.numValues()];

        for(int instEnum = 0; instEnum < att.numValues(); ++instEnum) {
            splitData[instEnum] = new Instances(data, data.numInstances());
        }

        Enumeration var6 = data.enumerateInstances();

        while(var6.hasMoreElements()) {
            Instance i = (Instance)var6.nextElement();
//            System.out.println(i);
//            System.out.println(i.value(att));
//            System.out.println((int)i.value(att));
            splitData[(int)i.value(att)].add(i);
        }
        
        

        for(int var7 = 0; var7 < splitData.length; ++var7) {
            splitData[var7].compactify();
        }
        return splitData;
    }


    private Instances[] partitionData(Instances data, int fold, int k){
    	Instances[] subsets = new Instances[2];
    	subsets[0] = new Instances(data, data.numInstances());
    	subsets[1] = new Instances(data, data.numInstances());
    	
    	
    	for (int i=0; i<data.numInstances(); i++) {
    		Instance s = data.instance(i);
    		if (i%fold==k) 			
    			subsets[0].add(s);
    		else 
    			subsets[1].add(s);
    	}
 	
    	subsets[0].compactify();
    	subsets[1].compactify();
    	
    	return subsets;
    }
    
    private String toString(int level) {
        StringBuffer text = new StringBuffer();
        if(this.m_Attribute == null) {
            if(Instance.isMissingValue(this.m_ClassValue)) {
                text.append(": null");
            } else {
                text.append(": " + this.m_ClassAttribute.value((int)this.m_ClassValue));
            }
        } else {
            for(int j = 0; j < this.m_Attribute.numValues(); ++j) {
                text.append("\n");

                for(int i = 0; i < level; ++i) {
                    text.append("|  ");
                }

                text.append(this.m_Attribute.name() + " = " + this.m_Attribute.value(j));
                text.append(this.m_Successors[j].toString(level + 1));
            }
        }
        //System.out.print(text);
        return text.toString();
    }
    
    private double computeAccuracy(Instances data) throws IOException, NoSupportForMissingValuesException {
    	double accuracy = 0.0;
    	int correct = 0;
    	for(int index =0; index<data.numInstances();index++) {
    		Instance testInstance = data.instance(index);
    		double prediction = classifyInstance(testInstance);
    		if (!Double.isNaN(prediction)) {
    			if (prediction == testInstance.classValue())
    				correct ++;
    		}
    	}
    	
    	accuracy = (double)correct / data.numInstances();
    	return accuracy;
    }

    private void printOutput(Instances data) throws IOException, NoSupportForMissingValuesException {
        FileWriter fStream = new FileWriter("output\\decision_tree\\decision-tree-output.txt");     // Output File
        BufferedWriter out = new BufferedWriter(fStream);
        for(int index =0; index<data.numInstances();index++) {
            Instance testRowInstance = data.instance(index);
            double prediction =classifyInstance(testRowInstance);

            if (Double.isNaN(prediction)){
            	out.write("NaN");
            }
            else 
            	out.write(data.classAttribute().value((int) prediction));
            out.newLine();
        }
        out.close();
    }
    
    //Classifies a given test instance using the decision tree.
    public double classifyInstance(Instance instance) throws NoSupportForMissingValuesException {
        if(instance.hasMissingValue()) {
            throw new NoSupportForMissingValuesException("DecisionTree: no missing values, please.");
        } else {
            return this.m_Attribute == null?this.m_ClassValue:this.m_Successors[(int)instance.value(this.m_Attribute)].classifyInstance(instance);
        }
    }

    public String toString() {
        return this.m_Distribution == null && this.m_Successors == null?"DecisionTree: No model built yet.":"DecisionTree\n\n" + this.toString(0);
    }
    
    public void decisionTree() throws Exception {
        //BufferedReader file = Utility.readFile("data\\decision_tree\\weather-nominal.arff");
        BufferedReader file = Utility.readFile("data\\decision_tree\\vote.arff");
        Instances data = new Instances(file);

        int cIdx=data.numAttributes()-1;
        data.setClassIndex(cIdx);
        
 
        int fold = 5;
        double accuracy = 0.0;
        for(int k=0; k<fold; k++) {
            Instances[] subsets = partitionData(data, fold, k);
            System.out.println(subsets[0].numInstances());
            System.out.println(subsets[1].numInstances());   
            
            buildClassifier(subsets[1]);
            String dtStr = this.toString();
            System.out.println(dtStr);
            printOutput(subsets[0]);
            accuracy += computeAccuracy(subsets[0]);
            
        }
        
        accuracy /= fold;
        System.out.println(accuracy);
        
//        Instances[] subsets = partitionData(data, 17, 0);
//        System.out.println(subsets[0].numInstances());
//        System.out.println(subsets[1].numInstances());   
        
//        buildClassifier(data);
//        String dtStr = this.toString();
//        System.out.println(dtStr);

        

        
        

    }
}

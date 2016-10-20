import java.io.IOException;

public class main {

	public static void main(String[] args) throws NumberFormatException, IOException {
		// TODO Auto-generated method stub
		int numoftopics = 4;
		int DocSize = 20;
		String wordDictFile = "data\\term_dict.txt";
		String docTermFile = "data\\CT.txt";
		Plsa myPLSA = new Plsa(numoftopics);
		myPLSA.setDocSize(DocSize);
		myPLSA.readWordDict(wordDictFile);
		myPLSA.readDocTermMatrix(docTermFile);
		int maxIter = 1000;
		myPLSA.train(maxIter);	
		double[][] theta = myPLSA.getDocTopics();
		double[][] beta = myPLSA.getTopicWordPros();
	}

}

/*
 * main.cpp
 *
 *  Created on: 12 mag 2017
 *      Author: angelo
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include <iostream>

#include <rapidxml.hpp>
#include <rapidxml_utils.hpp>

using namespace cv;
using namespace std;
using namespace rapidxml;

int DESCRIPTOR_TYPE = 1; // {0 = hog, 1 = lbp, 2 = bb, 3 = conc}
int LOAD_CLASSIFIER = 0;
int USE_MES = 0; // If MES is used, the DESCRIPTOR_TYPE and LOAD_CLASSIFIER vars are not considered

void convertVectorToMatrix(const vector<vector<float> > &hogResult, Mat &mat) {
	int featureVecSize = hogResult[0].size();

	for (unsigned int i = 0; i < hogResult.size(); i++) {
		for (int j = 0; j < featureVecSize; j++) {
			mat.at<float>(i, j) = hogResult[i][j];
		}
	}
}

void convertVectorToMatrix(const vector<Mat> &lbpResult, Mat &mat) {
	for (unsigned int i = 0; i < lbpResult.size(); i++) {
		for (int j = 0; j < lbpResult[i].cols; j++) {
			mat.at<float>(i, j) = lbpResult[i].at<float>(0, j);
		}
	}
}

void convertVectorToMatrix(const vector<int> &labels, Mat &labelsMat) {
	for (unsigned int i = 0; i < labels.size(); i++) {
		labelsMat.at<float>(i, 0) = labels[i];
	}
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy,
		Mat &testLabelsMat) {

	for (int i = 0; i < testResponse.rows; i++) {
		if (testResponse.at<float>(i, 0) == testLabelsMat.at<float>(i, 0)) {
			count = count + 1;
		}
	}
	accuracy = (count / testResponse.rows) * 100;
}

void SVMtrain(CvSVM &svm, Mat &featureVecMat, Mat &labelsMat) {
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	CvMat trainingDesc = featureVecMat;
	CvMat trainingLabels = labelsMat;
	svm.train(&trainingDesc, &trainingLabels, Mat(), Mat(), params);
	stringstream ss;
	ss << "svm_" << DESCRIPTOR_TYPE << "_classifier.xml";
	svm.save(ss.str().c_str());
}

void histogram(const Mat &src, Mat &hist, int numPatterns) {
	hist = Mat::zeros(1, numPatterns, CV_32FC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int bin = src.at<unsigned char>(i, j);
			hist.at<float>(0, bin) += 1;
		}
	}
}

void OLBP(const Mat &src, Mat &dst) {
	dst = Mat::zeros(src.rows - 2, src.cols - 2, CV_8UC1);
	for (int i = 1; i < src.rows - 1; i++) {
		for (int j = 1; j < src.cols - 1; j++) {
			float center = src.at<unsigned char>(i, j);
			unsigned char code = 0;
			code |= (src.at<unsigned char>(i - 1, j - 1) > center) << 7;
			code |= (src.at<unsigned char>(i - 1, j) > center) << 6;
			code |= (src.at<unsigned char>(i - 1, j + 1) > center) << 5;
			code |= (src.at<unsigned char>(i, j + 1) > center) << 4;
			code |= (src.at<unsigned char>(i + 1, j + 1) > center) << 3;
			code |= (src.at<unsigned char>(i + 1, j) > center) << 2;
			code |= (src.at<unsigned char>(i + 1, j - 1) > center) << 1;
			code |= (src.at<unsigned char>(i, j - 1) > center) << 0;
			dst.at<unsigned char>(i - 1, j - 1) = code;
		}
	}
}

void ELBP(const Mat &src, Mat &dst, int radius, int neighbors) {
	neighbors = max(min(neighbors, 31), 1); // set bounds
	dst = Mat::zeros(src.rows - 2 * radius, src.cols - 2 * radius, CV_32SC1);
	for (int n = 0; n < neighbors; n++) {
		// sample points
		float x = static_cast<float>(radius)
				* cos(2.0 * M_PI * n / static_cast<float>(neighbors));
		float y = static_cast<float>(radius)
				* -sin(2.0 * M_PI * n / static_cast<float>(neighbors));
		// relative indices
		int fx = static_cast<int>(floor(x));
		int fy = static_cast<int>(floor(y));
		int cx = static_cast<int>(ceil(x));
		int cy = static_cast<int>(ceil(y));
		// fractional part
		float ty = y - fy;
		float tx = x - fx;
		// set interpolation weights
		float w1 = (1 - tx) * (1 - ty);
		float w2 = tx * (1 - ty);
		float w3 = (1 - tx) * ty;
		float w4 = tx * ty;
		// iterate through your data
		for (int i = radius; i < src.rows - radius; i++) {
			for (int j = radius; j < src.cols - radius; j++) {
				float t = w1 * src.at<unsigned char>(i + fy, j + fx)
						+ w2 * src.at<unsigned char>(i + fy, j + cx)
						+ w3 * src.at<unsigned char>(i + cy, j + fx)
						+ w4 * src.at<unsigned char>(i + cy, j + cx);
				// we are dealing with floating point precision, so add some little tolerance
				dst.at<unsigned int>(i - radius, j - radius) += ((t
						> src.at<unsigned char>(i, j))
						&& (abs(t - src.at<unsigned char>(i, j))
								> std::numeric_limits<float>::epsilon())) << n;
			}
		}
	}
}

void computeHOG(Mat &featureVecMat, const vector<Mat> &img) {
	// The 2nd and 4th params are fixed. Choose 1st and 3th such that (1st-2nd)%3th = 0
	HOGDescriptor hog(Size(100, 100), Size(16, 16), Size(4, 4), Size(8, 8), 9,
			-1, 0.2, true, 64);
	vector<vector<float> > hogResult;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Convert image from BGR to Gray
		Mat grayImg;
		cvtColor(img[i], grayImg, COLOR_BGR2GRAY);
		// Calculate the feature vector of the gray image and push back it into the hogResult
		vector<float> featureVector;
		hog.compute(grayImg, featureVector);
		hogResult.push_back(featureVector);
	}

	// Converts the vector<vector<float>> into a Mat of float
	featureVecMat = Mat(hogResult.size(), hogResult[0].size(), CV_32FC1);
	convertVectorToMatrix(hogResult, featureVecMat);
}

void computeLBP(Mat &featureVecMat, const vector<Mat> &img) {
	vector<Mat> lbpResult;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Convert image from BGR to Gray
		Mat grayImg;
		cvtColor(img[i], grayImg, COLOR_BGR2GRAY);
		// Tiny bit of smoothing is always a good idea
		GaussianBlur(grayImg, grayImg, Size(7, 7), 5, 3, BORDER_CONSTANT);
		Mat lbp;
		ELBP(grayImg, lbp, 4, 4);
		//OLBP(img[i], lbp);
		normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);
		Mat hist;
		histogram(lbp, hist, 256); // 256 is the number of bins of the histogram. It changes with the neighbors
		lbpResult.push_back(hist);
	}

	// Converts the vector<Mat> into a Mat of float
	featureVecMat = Mat(lbpResult.size(), lbpResult[0].cols, CV_32FC1);
	convertVectorToMatrix(lbpResult, featureVecMat);
}

void countWhitePixels(const Mat matrix, int &whitePixels) {
	whitePixels = 0;
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			if (matrix.at<uchar>(i, j) == 255)
				whitePixels++;
		}
	}
}

// Mat3b bgr(Vec3b(100,24,90)) for example
void convertColorFromBGR2HSV(const Mat3b &bgr, Mat3b &hsv) {
	// Convert a BGR color to HSV
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
}

void computeBB(Mat &featureVecMat, const vector<Mat> &img) {
	vector<Mat> hsvImg;
	vector<vector<float> > dimensions;

	for (unsigned int i = 0; i < img.size(); i++) {
		// Apply Gaussian blur to remove some noise
		Mat im;
		GaussianBlur(img[i], im, Size(7, 7), 5, 3);

		stringstream ss;
		ss << "prova/" << i << "_original.jpg";
		imwrite(ss.str(), img[i]);

		// Convert the image from BGR to HSV
		Mat hsvImg;
		cvtColor(im, hsvImg, COLOR_BGR2HSV);

		// Define range of colors in HSV
		vector<vector<Scalar> > colors;

		// The colors are choosen by H-+10, S from minS to 255, V from minV to 255
		vector<Scalar> brown;
		Scalar lower_brown(3, 60, 95);
		Scalar upper_brown(23, 255, 255);
		brown.push_back(lower_brown);
		brown.push_back(upper_brown);

		vector<Scalar> green;
		Scalar lower_green(30, 20, 40);
		Scalar upper_green(80, 255, 255);
		green.push_back(lower_green);
		green.push_back(upper_green);

		vector<Scalar> gray;
		Scalar lower_gray(0, 0, 80);
		Scalar upper_gray(255, 50, 220);
		gray.push_back(lower_gray);
		gray.push_back(upper_gray);

		vector<Scalar> white;
		Scalar lower_white(0, 0, 235);
		Scalar upper_white(255, 5, 255);
		white.push_back(lower_white);
		white.push_back(upper_white);

		colors.push_back(brown);
		colors.push_back(green);
		colors.push_back(gray);
		colors.push_back(white);

		int maxWhitePixels = 0;
		Mat bestMask;
		for (unsigned int j = 0; j < colors.size(); j++) {
			// Threshold the HSV image to get only background of this color
			Mat mask;
			inRange(hsvImg, colors[j].at(0), colors[j].at(1), mask);

			int whitePixels;
			countWhitePixels(mask, whitePixels);
			if (whitePixels > maxWhitePixels) {
				maxWhitePixels = whitePixels;
				bestMask = mask;
			}
		}

		ss.str("");
		ss << "prova/" << i << "_bgmask" << ".jpg";
		imwrite(ss.str(), bestMask);

		// Invert the mask to get the object of interest
		bitwise_not(bestMask, bestMask);

		// Bitwise-AND mask and original image
		Mat res;
		bitwise_and(im, im, res, bestMask);

		ss.str("");
		ss << "prova/" << i << "_result" << ".jpg";
		imwrite(ss.str(), res);

		// Detect edges using Canny
		Mat canny_mat;
		Canny(res, canny_mat, 20, 70, 3, false);

		ss.str("");
		ss << "prova/" << i << "_edges" << ".jpg";
		imwrite(ss.str(), canny_mat);

		// Find contours
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(canny_mat, contours, hierarchy, CV_RETR_EXTERNAL,
				CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		// Merge all contours into one vector
		vector<Point> merged_contours_points;
		for (unsigned int j = 0; j < contours.size(); j++) {
			for (unsigned int k = 0; k < contours[j].size(); k++) {
				merged_contours_points.push_back(contours[j][k]);
			}
		}

		if (merged_contours_points.size() > 0) {
			// Merge all lines (contours) in one line through convex hull
			vector<Point> hull;
			convexHull(merged_contours_points, hull);

			// Get the rotated bounding box
			RotatedRect rotated_bounding = minAreaRect(hull);

			vector<float> dim;
			dim.push_back(rotated_bounding.size.width);
			dim.push_back(rotated_bounding.size.height);
			dimensions.push_back(dim);

			/*
			 // Draw the rotated bouding box
			 Mat drawing = Mat::zeros(canny_mat.size(), CV_8UC3);
			 Point2f rect_vertices[4];
			 rotated_bounding.points(rect_vertices);
			 for (int j = 0; j < 4; j++)
			 line(drawing, rect_vertices[j], rect_vertices[(j + 1) % 4],
			 Scalar(120, 200, 200), 1, 8);
			 imshow("Rotated bounding box", drawing);
			 waitKey(0);
			 */
		} else {
			vector<float> dim;
			dim.push_back(0);
			dim.push_back(0);
			dimensions.push_back(dim);
		}
	}
	// Converts the vector<vector<float>> into a Mat of float
	featureVecMat = Mat(dimensions.size(), dimensions[0].size(), CV_32FC1);
	convertVectorToMatrix(dimensions, featureVecMat);
}

void concatFeatureVectors(Mat &concatResult, const vector<Mat> &images) {
	Mat hogResult;
	computeHOG(hogResult, images);
	Mat lbpResult;
	computeLBP(lbpResult, images);
	Mat bbResult;
	computeBB(bbResult, images);

	Mat matArray[] = { hogResult, lbpResult, bbResult };
	hconcat(matArray, sizeof(matArray) / sizeof(*matArray), concatResult);
}

void loadLabels(vector<int> &labels, int pedNum, int vehNum, int unkNum) {
	// Atm 0 means Pedestrian label, 1 means Vehicles and 2 means Unknown
	for (int i = 0; i < (pedNum + vehNum + unkNum); i++) {
		if (i < pedNum)
			labels.push_back(0);
		else if (i >= pedNum && i < (pedNum + vehNum))
			labels.push_back(1);
		else if (i >= (pedNum + vehNum))
			labels.push_back(2);
	}
}

void loadImages(vector<Mat> &images, String pedPath, String vehPath,
		String unkPath) {
	vector<String> pedFilesNames;
	glob(pedPath, pedFilesNames, true);
	for (unsigned int i = 0; i < pedFilesNames.size(); i++) {
		Mat img = imread(pedFilesNames[i], CV_LOAD_IMAGE_COLOR);
		// Don't resize in the bounding box case
		//if (DESCRIPTOR_TYPE != 2)
		resize(img, img, Size(100, 100));
		images.push_back(img);
	}

	vector<String> vehFilesNames;
	glob(vehPath, vehFilesNames, true);
	for (unsigned int i = 0; i < vehFilesNames.size(); i++) {
		Mat img = imread(vehFilesNames[i], CV_LOAD_IMAGE_COLOR);
		// Don't resize in the bounding box case
		//if (DESCRIPTOR_TYPE != 2)
		resize(img, img, Size(100, 100));
		images.push_back(img);
	}

	vector<String> unkFilesNames;
	glob(unkPath, unkFilesNames, true);
	for (unsigned int i = 0; i < unkFilesNames.size(); i++) {
		Mat img = imread(unkFilesNames[i], CV_LOAD_IMAGE_COLOR);
		// Don't resize in the bounding box case
		//if (DESCRIPTOR_TYPE != 2)
		resize(img, img, Size(100, 100));
		images.push_back(img);
	}
}

void createLabelsMat(Mat &labelsMat, String pedPath, String vehPath,
		String unkPath) {
	// Loads all the samples names to calculate the number of samples
	vector<String> pedFilesNames, vehFilesNames, unkFilesNames;
	glob(pedPath, pedFilesNames, true);
	glob(vehPath, vehFilesNames, true);
	glob(unkPath, unkFilesNames, true);

	vector<int> labels;
	loadLabels(labels, pedFilesNames.size(), vehFilesNames.size(),
			unkFilesNames.size());
	// Converts the vector<int> of labels into a Mat (a column vector) of float
	labelsMat = Mat(labels.size(), 1, CV_32FC1);
	convertVectorToMatrix(labels, labelsMat);
}

void createFeatureVectorsMat(Mat &featureVecMat, String pedPath, String vehPath,
		String unkPath, bool isNotTraining) {
	// Loads samples and corresponding labels
	vector<Mat> images;
	loadImages(images, pedPath, vehPath, unkPath);

	switch (DESCRIPTOR_TYPE) {
	case 0: {
		// Computes HOG, calculating a matrix in which each row is a feature vector.
		computeHOG(featureVecMat, images);
	}
		break;
	case 1: {
		// Computes LBP, calculating a matrix in which each row is a feature vector.
		computeLBP(featureVecMat, images);
	}
		break;
	case 2: {
		// Computes the bounding boxes, calculating a matrix in which each row is a feature vector (width, height).
		computeBB(featureVecMat, images);
	}
		break;
	case 3: {
		// Computes the concatenation of different feature vectors (hog+lbp+bb).
		concatFeatureVectors(featureVecMat, images);
	}
		break;
	default:
		break;
	}
}

void computeMES() {
	// Load the 3 trained classifiers
	CvSVM svm_hog, svm_lbp, svm_bb;
	svm_hog.load("svm_0_classifier.xml");
	svm_lbp.load("svm_1_classifier.xml");
	svm_bb.load("svm_2_classifier.xml");

	// Create the matrix containing all the labels for the validation samples
	Mat validLabelsMat;
	createLabelsMat(validLabelsMat, "valid_pedestrians/*.jpg",
			"valid_vehicles/*.jpg", "valid_unknown/*.jpg");

	// This vector contains the 3 accuracies for hog, lbp and bb classifiers for the validation set
	vector<float> accuracies;
	for (int i = 0; i < 3; i++) {
		DESCRIPTOR_TYPE = i;

		// Create the matrix of all the feature vectors of validation samples and the matrix of all the labels of validation samples
		Mat validFeatureVecMat;
		createFeatureVectorsMat(validFeatureVecMat, "valid_pedestrians/*.jpg",
				"valid_vehicles/*.jpg", "valid_unknown/*.jpg", true);

		// Predict the validation samples class with the classifiers
		Mat testResponse;
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			svm_hog.predict(validFeatureVecMat, testResponse);
		}
			break;
		case 1: {
			svm_lbp.predict(validFeatureVecMat, testResponse);
		}
			break;
		case 2: {
			svm_bb.predict(validFeatureVecMat, testResponse);
		}
			break;
		default:
			break;
		}

		// Evaluate the classifier accuracy
		float count = 0;
		float accuracy = 0;
		SVMevaluate(testResponse, count, accuracy, validLabelsMat);

		accuracies.push_back(accuracy);
	}

	// Init the matrix of weightedResponse
	// The number of rows is equal to the total number of testing samples; the number of columns is equal to the number of classes
	vector<String> pedFilesNames, vehFilesNames, unkFilesNames;
	glob("test_pedestrians/*.jpg", pedFilesNames, true);
	glob("test_vehicles/*.jpg", vehFilesNames, true);
	glob("test_unknown/*.jpg", unkFilesNames, true);
	Mat weightedResponse = Mat::zeros(
			pedFilesNames.size() + vehFilesNames.size() + unkFilesNames.size(),
			3, CV_32FC1);

	// Create the matrix containing all the labels for the test samples
	Mat testLabelsMat;
	createLabelsMat(testLabelsMat, "test_pedestrians/*.jpg",
			"test_vehicles/*.jpg", "test_unknown/*.jpg");

	// Predict the testing samples labels with the 3 classifiers and weight the results with the accuracies
	for (int i = 0; i < 3; i++) {
		DESCRIPTOR_TYPE = i;

		// Create the matrix of all the feature vectors of testing samples and the matrix of all the labels of testing samples
		Mat testFeatureVecMat;
		createFeatureVectorsMat(testFeatureVecMat, "test_pedestrians/*.jpg",
				"test_vehicles/*.jpg", "test_unknown/*.jpg", true);

		// Predict the testing samples class with the classifiers
		Mat testResponse;
		switch (DESCRIPTOR_TYPE) {
		case 0: {
			svm_hog.predict(testFeatureVecMat, testResponse);
		}
			break;
		case 1: {
			svm_lbp.predict(testFeatureVecMat, testResponse);
		}
			break;
		case 2: {
			svm_bb.predict(testFeatureVecMat, testResponse);
		}
			break;
		default:
			break;
		}

		// Populate the weightedResponse matrix
		for (int j = 0; j < testResponse.rows; j++) {
			weightedResponse.at<float>(j, testResponse.at<float>(j, 0)) +=
					accuracies.at(i);
		}
	}

	// Calculate the final matrix of responses (1 column and n rows, where n is the number of testing samples) finding the max for each row
	Mat finalTestResponse(weightedResponse.rows, 1, CV_32FC1);
	for (int i = 0; i < weightedResponse.rows; i++) {
		float max = 0;
		for (int j = 0; j < weightedResponse.cols; j++) {
			if (max < weightedResponse.at<float>(i, j)) {
				max = weightedResponse.at<float>(i, j);
				finalTestResponse.at<float>(i, 0) = j;
			}
		}
	}

	// Evaluate the classifier accuracy
	float count = 0;
	float accuracy = 0;
	SVMevaluate(finalTestResponse, count, accuracy, testLabelsMat);

	// Print the result
	cout << "The accuracy is " << accuracy << "%" << endl;
}

void classify() {
	// If MES is not used
	if (!USE_MES) {
		CvSVM svm;

		if (!LOAD_CLASSIFIER) {
			// Create the matrix of all the feature vectors of training samples and the matrix of all the labels of training samples
			Mat featureVecMat, labelsMat;
			createFeatureVectorsMat(featureVecMat, "train_pedestrians/*.jpg",
					"train_vehicles/*.jpg", "train_unknown/*.jpg", false);
			createLabelsMat(labelsMat, "train_pedestrians/*.jpg",
					"train_vehicles/*.jpg", "train_unknown/*.jpg");
			// Train the SVM classifier
			SVMtrain(svm, featureVecMat, labelsMat);
		} else {
			stringstream ss;
			ss << "svm_" << DESCRIPTOR_TYPE << "_classifier.xml";
			svm.load(ss.str().c_str());
		}

		// Create the matrix of all the feature vectors of testing samples and the matrix of all the labels of testing samples
		Mat testFeatureVecMat, testLabelsMat;
		createFeatureVectorsMat(testFeatureVecMat, "test_pedestrians/*.jpg",
				"test_vehicles/*.jpg", "test_unknown/*.jpg", true);
		createLabelsMat(testLabelsMat, "test_pedestrians/*.jpg",
				"test_vehicles/*.jpg", "test_unknown/*.jpg");

		// Predict the testing samples class with the classifier
		Mat testResponse;
		svm.predict(testFeatureVecMat, testResponse);

		// Evaluate the classifier accuracy
		float count = 0;
		float accuracy = 0;
		SVMevaluate(testResponse, count, accuracy, testLabelsMat);

		// Print the result
		cout << "The accuracy is " << accuracy << "%" << endl;
	} else {
		// Use MES
		computeMES();
	}
}

void extractSamplesFromVideo(const String pathVideo, const String pathXml,
		const String pathSave) {
	// Open the video file
	VideoCapture cap(pathVideo);
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return;
	}

	// Parse the xml file into doc
	file<> xmlFile(pathXml.c_str());
	xml_document<> doc;
	doc.parse<0>(xmlFile.data());

	// Read the <sourcefile>...</sourcefile> node
	xml_node<> *sourcefileNode =
			doc.first_node()->first_node("data")->first_node();

	// Iterate through all <object>...</object> nodes
	for (xml_node<> *objNode = sourcefileNode->first_node("object"); objNode;
			objNode = objNode->next_sibling()) {
		// Read the 2nd <attribute>...</attribute> node inside <object>...</object>
		xml_node<> *attributeNode = objNode->first_node()->next_sibling();

		// Iterate through all <data:bbox>...</data:bbox> nodes
		for (xml_node<> *bboxNode = attributeNode->first_node("data:bbox");
				bboxNode; bboxNode = bboxNode->next_sibling()) {
			// Read the frame number of the bbox
			stringstream ss(bboxNode->first_attribute("framespan")->value());
			string framespan;
			getline(ss, framespan, ':');
			double frame = atof(framespan.c_str());

			// Read the width, height, x and y of the bbox
			int height = atoi(bboxNode->first_attribute("height")->value());
			int width = atoi(bboxNode->first_attribute("width")->value());
			int x = atoi(bboxNode->first_attribute("x")->value());
			int y = atoi(bboxNode->first_attribute("y")->value());

			// Set the next frame
			cap.set(CV_CAP_PROP_POS_FRAMES, frame);

			// Save the frame into a Mat (image)
			Mat frameImg;
			bool success = cap.read(frameImg);
			if (!success) {
				cout << "Cannot read  frame " << endl;
				continue;
			}

			// Check if the bbox goes out of the img
			if (x >= 0 && y >= 0 && (x + width) <= frameImg.cols
					&& (y + height) <= frameImg.rows) {
				// Create the Rect to crop the bbox from the original frame
				Rect roi(Point(x, y), Point(x + width, y + height));

				// Crop the bbox from the original frame
				Mat cropImage = frameImg(roi);

				// Save the bbox crop as jpeg file
				stringstream st;
				st << pathSave << frame << "_" << x << "_" << y << "_" << width
						<< "_" << height << ".jpg";
				imwrite(st.str(), cropImage);
			}
		}
	}
}

int main(int argc, char** argv) {
	//extractSamplesFromVideo("prova.mp4", "prova.xgtf", "prova/");
	classify();
	return (0);
}


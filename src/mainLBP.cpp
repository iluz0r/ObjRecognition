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

using namespace cv;
using namespace std;

void histogram(const Mat &src, Mat &hist, int numPatterns) {
	hist = Mat::zeros(1, numPatterns, CV_32SC1);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			int bin = src.at<unsigned char>(i, j);
			hist.at<int>(0, bin) += 1;
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

void loadImages(vector<Mat> &images, int &pedNum, int &vehiclesNum,
		String pedPath, String vehPath) {
	vector<String> pedFilesNames;
	glob(pedPath, pedFilesNames, true);
	for (unsigned int i = 0; i < pedFilesNames.size(); i++) {
		Mat img = imread(pedFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	pedNum = pedFilesNames.size();

	vector<String> vehFilesNames;
	glob(vehPath, vehFilesNames, true);
	for (unsigned int i = 0; i < vehFilesNames.size(); i++) {
		Mat img = imread(vehFilesNames[i], CV_LOAD_IMAGE_GRAYSCALE);
		Mat resizedImg;
		resize(img, resizedImg, Size(100, 100));
		images.push_back(resizedImg);
	}
	vehiclesNum = vehFilesNames.size();
}

int main(int argc, char** argv) {
	vector<Mat> trainImg;
	int trainPedNum, trainVehNum;
	loadImages(trainImg, trainPedNum, trainVehNum, "train_pedestrians/*.jpg",
			"train_vehicles/*.jpg");

	vector<Mat> trainLBP;
	for (unsigned int i = 0; i < trainImg.size(); i++) {
		Mat lbp;
		OLBP(trainImg[i], lbp);
		normalize(lbp, lbp, 0, 255, NORM_MINMAX, CV_8UC1);

		/*
		 stringstream ss;
		 ss << "LBP " << i;
		 namedWindow(ss.str(), CV_WINDOW_AUTOSIZE);
		 imshow(ss.str(), lbp);

		 stringstream ss2;
		 ss2 << "Original " << i;
		 namedWindow(ss2.str(), CV_WINDOW_AUTOSIZE);
		 imshow(ss2.str(), trainImg[i]);
		 waitKey(0);
		 */

		Mat hist;
		histogram(lbp, hist, 256);
		trainLBP.push_back(hist);
	}
	cout << trainLBP.size() << endl;
	return (0);
}

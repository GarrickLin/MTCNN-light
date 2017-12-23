#ifndef MTCNN_H
#define MTCNN_H
#include <memory>
#include <vector>
#include <opencv2/core.hpp>
#include "pBox.h"

using cv::Mat;
using std::vector;

class mtcnn{
public:
	mtcnn(int row, int col, int minsize = 60);
	~mtcnn();
	void findFace(Mat &image);
	int detectFace(const Mat &image, vector<Bbox>& facebox);
	int regressFace(const Mat &image, vector<Bbox>& facebox);
private:
	class impl;
	std::unique_ptr<impl> pimpl;
};

#endif
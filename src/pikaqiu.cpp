#include <opencv2/highgui.hpp>
#include "mtcnn.h"
#include <time.h>
using namespace cv;

#if 0
int main()
{
    //Mat image = imread("4.jpg");
    //mtcnn find(image.rows, image.cols);
    //clock_t start;
    //start = clock();
    //find.findFace(image);
    //imshow("result", image);
    //imwrite("result.jpg",image);
    //start = clock() -start;
    //cout<<"time is  "<<start/10e3<<endl;

	Mat image;
	VideoCapture cap(0);
	if (!cap.isOpened())
		cout << "fail to open!" << endl;
	cap >> image;
	if (!image.data){
		cout << "读取视频失败" << endl;
		return -1;
	}

	mtcnn find(image.rows, image.cols);
	clock_t start;
	int stop = 1200;
	while (1){
		start = clock();
		cap >> image;
		find.findFace(image);
		imshow("result", image);
		if (waitKey(1) >= 0) break;
		start = clock() - start;
		cout << "time is  " << start / 10e3 << endl;
	}
    image.release();
    return 0;
}


#else

int main(){
	Mat image;
	VideoCapture cap(0);
	if (!cap.isOpened())
		cout << "fail to open!" << endl;
	cap >> image;
	if (!image.data){
		cout << "读取视频失败" << endl;
		return -1;
	}

	mtcnn find(image.rows, image.cols);
	vector<Bbox> facebox;
	clock_t start;
	int num = 0;
	while (1){
		cap >> image;
		start = clock();
		if (num){
			num = find.regressFace(image, facebox);
			printf("tracking by regression, number of faces %d\n", num);
		}
		else{
			num = find.detectFace(image, facebox);
			printf("detecting number of faces %d\n", num);
		}
		start = clock() - start;
		cout << "time is  " << start / 1e3 << endl;
		// render
		for (vector<Bbox>::iterator it = facebox.begin(); it != facebox.end(); it++){
			rectangle(image, Point(it->y1, it->x1), Point(it->y2, it->x2), Scalar(0, 0, 255), 2, 8, 0);
			for (int num = 0; num < 5; num++)
				circle(image, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
		}
		imshow("result", image);
		if (waitKey(1) >= 0) break;

	}

	image.release();
	return 0;
}

#endif
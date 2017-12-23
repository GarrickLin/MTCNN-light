#include "mtcnn.h"
#include "network.h"
#include "pBox.h"

using namespace cv;
using std::cout;
using std::endl;

class Pnet
{
public:
	Pnet();
	~Pnet();
	void run(Mat &image, float scale);

	float nms_threshold;
	mydataFmt Pthreshold;
	bool firstFlag;
	vector<struct Bbox> boundingBox_;
	vector<orderScore> bboxScore_;
private:
	//the image for mxnet conv
	struct pBox *rgb;
	struct pBox *conv1_matrix;
	//the 1th layer's out conv
	struct pBox *conv1;
	struct pBox *maxPooling1;
	struct pBox *maxPooling_matrix;
	//the 3th layer's out
	struct pBox *conv2;
	struct pBox *conv3_matrix;
	//the 4th layer's out   out
	struct pBox *conv3;
	struct pBox *score_matrix;
	//the 4th layer's out   out
	struct pBox *score_;
	//the 4th layer's out   out
	struct pBox *location_matrix;
	struct pBox *location_;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *conv4c1_wb;
	struct Weight *conv4c2_wb;

	void generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale);
};

class Rnet
{
public:
	Rnet();
	~Rnet();
	float Rthreshold;
	void run(Mat &image);
	struct pBox *score_;
	struct pBox *location_;
private:
	struct pBox *rgb;

	struct pBox *conv1_matrix;
	struct pBox *conv1_out;
	struct pBox *pooling1_out;

	struct pBox *conv2_matrix;
	struct pBox *conv2_out;
	struct pBox *pooling2_out;

	struct pBox *conv3_matrix;
	struct pBox *conv3_out;

	struct pBox *fc4_out;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *fc4_wb;
	struct pRelu *prelu_gmma4;
	struct Weight *score_wb;
	struct Weight *location_wb;

	void RnetImage2MatrixInit(struct pBox *pbox);
};

class Onet
{
public:
	Onet();
	~Onet();
	void run(Mat &image);
	float Othreshold;
	struct pBox *score_;
	struct pBox *location_;
	struct pBox *keyPoint_;
private:
	struct pBox *rgb;
	struct pBox *conv1_matrix;
	struct pBox *conv1_out;
	struct pBox *pooling1_out;

	struct pBox *conv2_matrix;
	struct pBox *conv2_out;
	struct pBox *pooling2_out;

	struct pBox *conv3_matrix;
	struct pBox *conv3_out;
	struct pBox *pooling3_out;

	struct pBox *conv4_matrix;
	struct pBox *conv4_out;

	struct pBox *fc5_out;

	//Weight
	struct Weight *conv1_wb;
	struct pRelu *prelu_gmma1;
	struct Weight *conv2_wb;
	struct pRelu *prelu_gmma2;
	struct Weight *conv3_wb;
	struct pRelu *prelu_gmma3;
	struct Weight *conv4_wb;
	struct pRelu *prelu_gmma4;
	struct Weight *fc5_wb;
	struct pRelu *prelu_gmma5;
	struct Weight *score_wb;
	struct Weight *location_wb;
	struct Weight *keyPoint_wb;
	void OnetImage2MatrixInit(struct pBox *pbox);
};

class mtcnn::impl
{
public:
	impl(int row, int col, int minsize = 60);
	~impl();
	void findFace(Mat &image);
	int detectFace(const Mat &image, vector<Bbox>& facebox);
	int regressFace(const Mat &image, vector<Bbox>& facebox);
private:
	Mat reImage;
	float nms_threshold[3];
	vector<float> scales_;
	Pnet *simpleFace_;
	vector<struct Bbox> firstBbox_;
	vector<struct orderScore> firstOrderScore_;
	Rnet refineNet;
	vector<struct Bbox> secondBbox_;
	vector<struct orderScore> secondBboxScore_;
	Onet outNet;
	vector<struct Bbox> thirdBbox_;
	vector<struct orderScore> thirdBboxScore_;
};

Pnet::Pnet(){
    Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;
    this->rgb = new pBox;

    this->conv1_matrix = new pBox;
    this->conv1 = new pBox;
    this->maxPooling1 = new pBox;

    this->maxPooling_matrix = new pBox;
    this->conv2 = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3 = new pBox;

    this->score_matrix = new pBox;
    this->score_ = new pBox;

    this->location_matrix = new pBox;
    this->location_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
    this->conv4c1_wb = new Weight;
    this->conv4c2_wb = new Weight;
    //                                 w       sc  lc ks s  p
    long conv1 = initConvAndFc(this->conv1_wb, 10, 3, 3, 1, 0);
    initpRelu(this->prelu_gmma1, 10);
    long conv2 = initConvAndFc(this->conv2_wb, 16, 10, 3, 1, 0);
    initpRelu(this->prelu_gmma2, 16);
    long conv3 = initConvAndFc(this->conv3_wb, 32, 16, 3, 1, 0);
    initpRelu(this->prelu_gmma3, 32);
    long conv4c1 = initConvAndFc(this->conv4c1_wb, 2, 32, 1, 1, 0);
    long conv4c2 = initConvAndFc(this->conv4c2_wb, 4, 32, 1, 1, 0);
    long dataNumber[13] = {conv1,10,10, conv2,16,16, conv3,32,32, conv4c1,2, conv4c2,4};
    mydataFmt *pointTeam[13] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                            this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                            this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                            this->conv4c1_wb->pdata, this->conv4c1_wb->pbias, \
                            this->conv4c2_wb->pdata, this->conv4c2_wb->pbias \
                            };
    string filename = "Pnet.txt";
    readData(filename, dataNumber, pointTeam);
}
Pnet::~Pnet(){
    freepBox(this->rgb);
    freepBox(this->conv1);
    freepBox(this->maxPooling1);
    freepBox(this->conv2);
    freepBox(this->conv3);
    freepBox(this->score_);
    freepBox(this->location_);

    freepBox(this->conv1_matrix);
    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freepBox(this->maxPooling_matrix);
    freeWeight(this->conv2_wb);
    freepBox(this->conv3_matrix);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepBox(this->score_matrix);
    freepRelu(this->prelu_gmma3);
    freeWeight(this->conv4c1_wb);
    freepBox(this->location_matrix);
    freeWeight(this->conv4c2_wb);
}

void Pnet::run(Mat &image, float scale){
    if(firstFlag){
        image2MatrixInit(image, this->rgb);

        feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
        convolutionInit(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);

        maxPoolingInit(this->conv1, this->maxPooling1, 2, 2);
        feature2MatrixInit(this->maxPooling1, this->maxPooling_matrix, this->conv2_wb);
        convolutionInit(this->conv2_wb, this->maxPooling1, this->conv2, this->maxPooling_matrix);
        
        feature2MatrixInit(this->conv2, this->conv3_matrix, this->conv3_wb);
        convolutionInit(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);

        feature2MatrixInit(this->conv3, this->score_matrix, this->conv4c1_wb);
        convolutionInit(this->conv4c1_wb, this->conv3, this->score_, this->score_matrix);

        feature2MatrixInit(this->conv3, this->location_matrix, this->conv4c2_wb);
        convolutionInit(this->conv4c2_wb, this->conv3, this->location_, this->location_matrix);
        firstFlag = false;
    }

    image2Matrix(image, this->rgb);

    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);
    prelu(this->conv1, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
    //Pooling layer
    maxPooling(this->conv1, this->maxPooling1, 2, 2);

    feature2Matrix(this->maxPooling1, this->maxPooling_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->maxPooling1, this->conv2, this->maxPooling_matrix);
    prelu(this->conv2, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    //conv3 
    feature2Matrix(this->conv2, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);
    prelu(this->conv3, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
    //conv4c1   score
    feature2Matrix(this->conv3, this->score_matrix, this->conv4c1_wb);
    convolution(this->conv4c1_wb, this->conv3, this->score_, this->score_matrix);
    addbias(this->score_, this->conv4c1_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);

    //conv4c2   location
    feature2Matrix(this->conv3, this->location_matrix, this->conv4c2_wb);
    convolution(this->conv4c2_wb, this->conv3, this->location_, this->location_matrix);
    addbias(this->location_, this->conv4c2_wb->pbias);
    //softmax layer
    generateBbox(this->score_, this->location_, scale);
}
void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale){
    //for pooling 
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = score->pdata + score->width*score->height;
    mydataFmt *plocal = location->pdata;
    struct Bbox bbox;
    struct orderScore order;
    for(int row=0;row<score->height;row++){
        for(int col=0;col<score->width;col++){
            if(*p>Pthreshold){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*row+1)/scale);
                bbox.y1 = round((stride*col+1)/scale);
                bbox.x2 = round((stride*row+1+cellsize)/scale);
                bbox.y2 = round((stride*col+1+cellsize)/scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal+channel*location->width*location->height);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}

Rnet::Rnet(){
    Rthreshold = 0.7;

    this->rgb = new pBox;
    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;
    this->pooling1_out = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;
    this->pooling2_out = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;

    this->fc4_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
    this->fc4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;
    this->score_wb = new Weight;
    this->location_wb = new Weight;
    // //                             w         sc  lc ks s  p
    long conv1 = initConvAndFc(this->conv1_wb, 28, 3, 3, 1, 0);
    initpRelu(this->prelu_gmma1, 28);
    long conv2 = initConvAndFc(this->conv2_wb, 48, 28, 3, 1, 0);
    initpRelu(this->prelu_gmma2, 48);
    long conv3 = initConvAndFc(this->conv3_wb, 64, 48, 2, 1, 0);
    initpRelu(this->prelu_gmma3, 64);
    long fc4 = initConvAndFc(this->fc4_wb, 128, 576, 1, 1, 0);
    initpRelu(this->prelu_gmma4, 128);
    long score = initConvAndFc(this->score_wb, 2, 128, 1, 1, 0);
    long location = initConvAndFc(this->location_wb, 4, 128, 1, 1, 0);
    long dataNumber[16] = {conv1,28,28, conv2,48,48, conv3,64,64, fc4,128,128, score,2, location,4};
    mydataFmt *pointTeam[16] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->fc4_wb->pdata, this->fc4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias \
                                };
    string filename = "Rnet.txt";
    readData(filename, dataNumber, pointTeam);

    //Init the network
    RnetImage2MatrixInit(rgb);
    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    maxPoolingInit(this->conv1_out, this->pooling1_out, 3, 2);
    feature2MatrixInit(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    maxPoolingInit(this->conv2_out, this->pooling2_out, 3, 2);
    feature2MatrixInit(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    fullconnectInit(this->fc4_wb, this->fc4_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Rnet::~Rnet(){
    freepBox(this->rgb);
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
    freepBox(this->pooling1_out);
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
    freepBox(this->pooling2_out);
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
    freepBox(this->fc4_out);
    freepBox(this->score_);
    freepBox(this->location_);

    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);
    freeWeight(this->fc4_wb);
    freepRelu(this->prelu_gmma4);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
}
void Rnet::RnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 24;
    pbox->width = 24;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL) cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Rnet::run(Mat &image){
    image2Matrix(image, this->rgb);

    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);

    maxPooling(this->conv1_out, this->pooling1_out, 3, 2);

    feature2Matrix(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    maxPooling(this->conv2_out, this->pooling2_out, 3, 2);

    //conv3 
    feature2Matrix(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);

    //flatten
    fullconnect(this->fc4_wb, this->conv3_out, this->fc4_out);
    prelu(this->fc4_out, this->fc4_wb->pbias, this->prelu_gmma4->pdata);

    //conv51   score
    fullconnect(this->score_wb, this->fc4_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);

    //conv5_2   location
    fullconnect(this->location_wb, this->fc4_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);
}

Onet::Onet(){
    Othreshold = 0.8;
    this->rgb = new pBox;

    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;
    this->pooling1_out = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;
    this->pooling2_out = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;
    this->pooling3_out = new pBox;

    this->conv4_matrix = new pBox;
    this->conv4_out = new pBox;

    this->fc5_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;
    this->keyPoint_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
    this->conv4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;
    this->fc5_wb = new Weight;
    this->prelu_gmma5 = new pRelu;
    this->score_wb = new Weight;
    this->location_wb = new Weight;
    this->keyPoint_wb = new Weight;

    // //                             w        sc  lc ks s  p
    long conv1 = initConvAndFc(this->conv1_wb, 32, 3, 3, 1, 0);
    initpRelu(this->prelu_gmma1, 32);
    long conv2 = initConvAndFc(this->conv2_wb, 64, 32, 3, 1, 0);
    initpRelu(this->prelu_gmma2, 64);
    long conv3 = initConvAndFc(this->conv3_wb, 64, 64, 3, 1, 0);
    initpRelu(this->prelu_gmma3, 64);
    long conv4 = initConvAndFc(this->conv4_wb, 128, 64, 2, 1, 0);
    initpRelu(this->prelu_gmma4, 128);
    long fc5 = initConvAndFc(this->fc5_wb, 256, 1152, 1, 1, 0);
    initpRelu(this->prelu_gmma5, 256);
    long score = initConvAndFc(this->score_wb, 2, 256, 1, 1, 0);
    long location = initConvAndFc(this->location_wb, 4, 256, 1, 1, 0);
    long keyPoint = initConvAndFc(this->keyPoint_wb, 10, 256, 1, 1, 0);
    long dataNumber[21] = {conv1,32,32, conv2,64,64, conv3,64,64, conv4,128,128, fc5,256,256, score,2, location,4, keyPoint,10};
    mydataFmt *pointTeam[21] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->fc5_wb->pdata, this->fc5_wb->pbias, this->prelu_gmma5->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias, \
                                this->keyPoint_wb->pdata, this->keyPoint_wb->pbias \
                                };
    string filename = "Onet.txt";
    readData(filename, dataNumber, pointTeam);

    //Init the network
    OnetImage2MatrixInit(rgb);

    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    maxPoolingInit(this->conv1_out, this->pooling1_out, 3, 2);

    feature2MatrixInit(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    maxPoolingInit(this->conv2_out, this->pooling2_out, 3, 2);

    feature2MatrixInit(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    maxPoolingInit(this->conv3_out, this->pooling3_out, 2, 2);

    feature2MatrixInit(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolutionInit(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);

    fullconnectInit(this->fc5_wb, this->fc5_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
    fullconnectInit(this->keyPoint_wb, this->keyPoint_);
}
Onet::~Onet(){
    freepBox(this->rgb);
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
    freepBox(this->pooling1_out);
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
    freepBox(this->pooling2_out);
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
    freepBox(this->pooling3_out);
    freepBox(this->conv4_matrix);
    freepBox(this->conv4_out);
    freepBox(this->fc5_out);
    freepBox(this->score_);
    freepBox(this->location_);
    freepBox(this->keyPoint_);

    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);
    freeWeight(this->conv4_wb);
    freepRelu(this->prelu_gmma4);
    freeWeight(this->fc5_wb);
    freepRelu(this->prelu_gmma5);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
    freeWeight(this->keyPoint_wb);
}
void Onet::OnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 48;
    pbox->width = 48;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL) cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Onet::run(Mat &image){
    image2Matrix(image, this->rgb);

    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);

    //Pooling layer
    maxPooling(this->conv1_out, this->pooling1_out, 3, 2);

    feature2Matrix(this->pooling1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->pooling1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    maxPooling(this->conv2_out, this->pooling2_out, 3, 2);

    //conv3 
    feature2Matrix(this->pooling2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->pooling2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
    maxPooling(this->conv3_out, this->pooling3_out, 2, 2);

    //conv4
    feature2Matrix(this->pooling3_out, this->conv4_matrix, this->conv4_wb);
    convolution(this->conv4_wb, this->pooling3_out, this->conv4_out, this->conv4_matrix);
    prelu(this->conv4_out, this->conv4_wb->pbias, this->prelu_gmma4->pdata);

    fullconnect(this->fc5_wb, this->conv4_out, this->fc5_out);
    prelu(this->fc5_out, this->fc5_wb->pbias, this->prelu_gmma5->pdata);

    //conv6_1   score
    fullconnect(this->score_wb, this->fc5_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);

    //conv6_2   location
    fullconnect(this->location_wb, this->fc5_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);

    //conv6_2   location
    fullconnect(this->keyPoint_wb, this->fc5_out, this->keyPoint_);
    addbias(this->keyPoint_, this->keyPoint_wb->pbias);
    // pBoxShow(keyPoint_);
}


mtcnn::impl::impl(int row, int col, int minsize){
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.7;
    nms_threshold[2] = 0.7;
	minsize = std::max(minsize, 12);
    float minl = row>col?col:row;
    int MIN_DET_SIZE = 12;
    float m = (float)MIN_DET_SIZE/minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;

    while(minl>MIN_DET_SIZE){
        if(factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    float minside = row<col ? row : col;
    int count = 0;
    for (vector<float>::iterator it = scales_.begin(); it != scales_.end(); it++){
        if (*it > 1){
			throw std::invalid_argument("the minsize is too small");
        }
        if (*it < (MIN_DET_SIZE / minside)){
            scales_.resize(count);
            break;
        }
        count++;
    }
    simpleFace_ = new Pnet[scales_.size()];
}

mtcnn::impl::~impl(){
    delete []simpleFace_;
}

void mtcnn::impl::findFace(Mat &image){
    struct orderScore order;
    int count = 0;
	if (image.empty())
		return;
    for (size_t i = 0; i < scales_.size(); i++) {
        int changedH = (int)ceil(image.rows*scales_.at(i));
        int changedW = (int)ceil(image.cols*scales_.at(i));
        resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
        simpleFace_[i].run(reImage, scales_.at(i));
        nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

        for(vector<struct Bbox>::iterator it=simpleFace_[i].boundingBox_.begin(); it!=simpleFace_[i].boundingBox_.end();it++){
            if((*it).exist){
                firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                firstOrderScore_.push_back(order);
                count++;
            }
        }
        simpleFace_[i].bboxScore_.clear();
        simpleFace_[i].boundingBox_.clear();
    }
    //the first stage's nms
    if(count<1)return;
    nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
    refineAndSquareBbox(firstBbox_, image.rows, image.cols);

    //second stage
    count = 0;
    for(vector<struct Bbox>::iterator it=firstBbox_.begin(); it!=firstBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat secImage;
			if (0 <= temp.x && temp.x<temp.x+temp.width && temp.x+temp.width<=image.cols &&
				0 <= temp.y && temp.y<temp.y+temp.height && temp.y+temp.height<=image.rows){
				resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
			}
			else{
				continue;
			}
            refineNet.run(secImage);
            if(*(refineNet.score_->pdata+1)>refineNet.Rthreshold){
                memcpy(it->regreCoord, refineNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(refineNet.score_->pdata+1);
                secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                secondBboxScore_.push_back(order);
            }
            else{
                (*it).exist=false;
            }
        }
    }
    if(count<1)return;
    nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
    refineAndSquareBbox(secondBbox_, image.rows, image.cols);

    //third stage 
    count = 0;
    for(vector<struct Bbox>::iterator it=secondBbox_.begin(); it!=secondBbox_.end();it++){
        if((*it).exist){
            Rect temp((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
            Mat thirdImage;
			if (0 <= temp.x && temp.x < temp.x + temp.width && temp.x + temp.width <= image.cols &&
				0 <= temp.y && temp.y<temp.y + temp.height && temp.y + temp.height <= image.rows){
				resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
			}
			else{
				continue;
			}            
            outNet.run(thirdImage);
            mydataFmt *pp=NULL;
            if(*(outNet.score_->pdata+1)>outNet.Othreshold){
                memcpy(it->regreCoord, outNet.location_->pdata, 4*sizeof(mydataFmt));
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = *(outNet.score_->pdata+1);
                pp = outNet.keyPoint_->pdata;
                for(int num=0;num<5;num++){
                    (it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp+num));
                }
                for(int num=0;num<5;num++){
                    (it->ppoint)[num+5] = it->x1 + (it->x2 - it->x1)*(*(pp+num+5));
                }
                thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                thirdBboxScore_.push_back(order);
            }
            else{
                it->exist=false;
            }
        }
    }

    if(count<1)return;
    refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
    nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
    for(vector<struct Bbox>::iterator it=thirdBbox_.begin(); it!=thirdBbox_.end();it++){
        if((*it).exist){
            rectangle(image, Point((*it).y1, (*it).x1), Point((*it).y2, (*it).x2), Scalar(0,0,255), 2,8,0);
            for(int num=0;num<5;num++)circle(image,Point((int)*(it->ppoint+num), (int)*(it->ppoint+num+5)),3,Scalar(0,255,255), -1);
        }
    }
    firstBbox_.clear();
    firstOrderScore_.clear();
    secondBbox_.clear();
    secondBboxScore_.clear();
    thirdBbox_.clear();
    thirdBboxScore_.clear();
}


int mtcnn::impl::detectFace(const Mat &image, vector<struct Bbox>& facebox){
	facebox.clear();
	if (image.empty())
		return 0;
	struct orderScore order;
	int count = 0;
	for (size_t i = 0; i < scales_.size(); i++) {
		int changedH = (int)ceil(image.rows*scales_.at(i));
		int changedW = (int)ceil(image.cols*scales_.at(i));
		resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
		simpleFace_[i].run(reImage, scales_.at(i));
		nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

		for (vector<struct Bbox>::iterator it = simpleFace_[i].boundingBox_.begin(); it != simpleFace_[i].boundingBox_.end(); it++){
			if ((*it).exist){
				firstBbox_.push_back(*it);
				order.score = (*it).score;
				order.oriOrder = count;
				firstOrderScore_.push_back(order);
				count++;
			}
		}
		simpleFace_[i].bboxScore_.clear();
		simpleFace_[i].boundingBox_.clear();
	}
	//the first stage's nms
	if (count<1)
		return count;
	nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
	refineAndSquareBbox(firstBbox_, image.rows, image.cols);

	//second stage
	count = 0;
	for (vector<struct Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).y1, (*it).x1, (*it).y2 - (*it).y1, (*it).x2 - (*it).x1);
			Mat secImage;
			if (0 <= temp.x && temp.x < temp.x + temp.width && temp.x + temp.width <= image.cols &&
				0 <= temp.y && temp.y<temp.y + temp.height && temp.y + temp.height <= image.rows){
				resize(image(temp), secImage, Size(24, 24), 0, 0, cv::INTER_LINEAR);
			}
			else{
				continue;
			}
			refineNet.run(secImage);
			if (*(refineNet.score_->pdata + 1)>refineNet.Rthreshold){
				memcpy(it->regreCoord, refineNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(refineNet.score_->pdata + 1);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
			}
			else{
				(*it).exist = false;
			}
		}
	}
	if (count<1)
		return count;
	nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
	refineAndSquareBbox(secondBbox_, image.rows, image.cols);

	//third stage 
	count = 0;
	for (vector<struct Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).y1, (*it).x1, (*it).y2 - (*it).y1, (*it).x2 - (*it).x1);
			Mat thirdImage;
			if (0 <= temp.x && temp.x < temp.x + temp.width && temp.x + temp.width <= image.cols &&
				0 <= temp.y && temp.y<temp.y + temp.height && temp.y + temp.height <= image.rows){
				resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
			}
			else{
				continue;
			}
			outNet.run(thirdImage);
			mydataFmt *pp = NULL;
			if (*(outNet.score_->pdata + 1)>outNet.Othreshold){
				memcpy(it->regreCoord, outNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(outNet.score_->pdata + 1);
				pp = outNet.keyPoint_->pdata;
				for (int num = 0; num < 5; num++){
					(it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp + num));
				}
				for (int num = 0; num < 5; num++){
					(it->ppoint)[num + 5] = it->x1 + (it->x2 - it->x1)*(*(pp + num + 5));
				}
				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			else{
				it->exist = false;
			}
		}
	}

	if (count < 1)
		return count;
	refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
	nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
	// set to output vector	
	for (vector<struct Bbox>::iterator it = thirdBbox_.begin(); it != thirdBbox_.end(); it++){
		if (it->exist){
			facebox.push_back((*it));
		}
	}
	firstBbox_.clear();
	firstOrderScore_.clear();
	secondBbox_.clear();
	secondBboxScore_.clear();
	thirdBbox_.clear();
	thirdBboxScore_.clear();
	return count;
}

int mtcnn::impl::regressFace(const Mat &image, vector<Bbox>& facebox){
	struct orderScore order;
	float regthreshold = 0.8;
	//third stage 
	int count = 0;
	for (vector<struct Bbox>::iterator it = facebox.begin(); it != facebox.end(); it++){
		if ((*it).exist){
			Rect temp((*it).y1, (*it).x1, (*it).y2 - (*it).y1, (*it).x2 - (*it).x1);
			Mat thirdImage;
			if (0 <= temp.x && temp.x < temp.x + temp.width && temp.x + temp.width <= image.cols &&
				0 <= temp.y && temp.y<temp.y + temp.height && temp.y + temp.height <= image.rows){
				resize(image(temp), thirdImage, Size(48, 48), 0, 0, cv::INTER_LINEAR);
			}
			else{
				continue;
			}
			outNet.run(thirdImage);
			mydataFmt *pp = NULL;
			if (*(outNet.score_->pdata + 1) > regthreshold){
				memcpy(it->regreCoord, outNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(outNet.score_->pdata + 1);
				pp = outNet.keyPoint_->pdata;
				for (int num = 0; num < 5; num++){
					(it->ppoint)[num] = it->y1 + (it->y2 - it->y1)*(*(pp + num));
				}
				for (int num = 0; num < 5; num++){
					(it->ppoint)[num + 5] = it->x1 + (it->x2 - it->x1)*(*(pp + num + 5));
				}
				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			else{
				it->exist = false;
			}
		}
	}

	if (count < 1) return count;
	refineAndSquareBbox(thirdBbox_, image.rows, image.cols);
	if (count > 1)
		nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");
	// set to output vector	
	facebox.clear();
	for (vector<struct Bbox>::iterator it = thirdBbox_.begin(); it != thirdBbox_.end(); it++){
		if (it->exist){
			facebox.push_back((*it));
		}
	}
	thirdBbox_.clear();
	thirdBboxScore_.clear();
	return count;
}

mtcnn::mtcnn(int row, int col, int minsize /* = 60 */){
	pimpl = std::make_unique<impl>(row, col, minsize);
}

mtcnn::~mtcnn(){}

void mtcnn::findFace(Mat &image){
	pimpl->findFace(image);
}

int mtcnn::detectFace(const Mat &image, vector<Bbox>& facebox){

	return pimpl->detectFace(image, facebox);
}

int mtcnn::regressFace(const Mat &image, vector<Bbox>& facebox){
	return pimpl->regressFace(image, facebox);
}
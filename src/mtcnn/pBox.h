#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>
#include "bbox.h"

struct pBox
{
	mydataFmt *pdata;
	int width;
	int height;
	int channel;
};

struct pRelu
{
    mydataFmt *pdata;
    int width;
};

struct Weight
{
	mydataFmt *pdata;
    mydataFmt *pbias;
    int lastChannel;
    int selfChannel;
	int kernelSize;
    int stride;
    int pad;
};

struct orderScore
{
    mydataFmt score;
    int oriOrder;
};

void freepBox(struct pBox *pbox);
void freeWeight(struct Weight *weight);
void freepRelu(struct pRelu *prelu);
void pBoxShow(const struct pBox *pbox);
void pBoxShowE(const struct pBox *pbox,int channel, int row);
void weightShow(const struct Weight *weight);
void pReluShow(const struct pRelu *prelu);

#endif
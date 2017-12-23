#pragma once
#define mydataFmt float

struct Bbox
{
	float score;
	int x1;
	int y1;
	int x2;
	int y2;
	float area;
	bool exist;
	mydataFmt ppoint[10];
	mydataFmt regreCoord[4];
};
#ifndef HULL_H
#define HULL_H

#include "point.h"
#include <point3d.h>
#include <vector>


class Hull{


public:
	static std::vector<Shared_Point> alphaShape2D(const std::list<Point> points);
	static std::list< std::pair<Shared_Point,Shared_Point> > alphaHull2D(const std::list<Point> points);
	static std::vector<Shared_Point> sortAlphaShapePoints(const std::vector<Shared_Point> hull_points);
	static std::list < std::list< std::pair<Shared_Point,Shared_Point> > > sortedPair(const std::list<Point> points);
	static std::vector<Shared_Point> sortFromAngle(const std::vector<Shared_Point> hull_points);
	static bool angleSorter(const Shared_Point p1, const Shared_Point p2);
	
};


#endif 

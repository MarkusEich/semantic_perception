#ifndef OBJECT_DETECTION_OBJECTS_HPP__
#define OBJECT_DETECTION_OBJECTS_HPP__

#include <base/eigen.h>
#include <boost/shared_ptr.hpp>
#include <stdexcept>
#include <inttypes.h>

namespace object_detection
{

/*
 * @brief specifies the color of an object or surface
 *
 * For now this is very simple, and internally holds an rgb representation of
 * the color.  In reality this is of course much more complicated since
 * different color profiles of the sensor/display devices have to be taken into
 * account. So the representation can be considered canonical.
 *
 * @todo once this class is matured, move it into base/types.
 */ 
struct Color
{
    /** red channel */
    float r;
    /** green channel */
    float g;
    /** blue channel */
    float b;

    /** default constructor returns black */
    Color() : r(.0f), g(.0f), b(.0f) {}

    /** construct based on rgb values */
    Color( float r, float g, float b )
	: r( r ), g( g ), b( b ) {}

    /** create Color from from 8bit RGB
     */
    static Color fromRGB8( uint8_t r, uint8_t g, uint8_t b )
    {
	const float max = 255.0f;
	return Color( r/max, g/max, b/max );
    }

    static Color fromHSV( float h, float s, float v )
    {
	throw std::runtime_error("not implemented");
    }

    /** to RGB color space
     *
     * no conversion is needed since this is the canonical representation.
     */
    void toRGB( float &r, float &g, float &b )
    {
	r = this->r;
	g = this->g;
	b = this->b;
    }

    void toHSV( float &h, float &s, float &v )
    {
	throw std::runtime_error("not implemented");
    }

    /** to RGB color space
     *
     * no conversion is needed since this is the canonical representation.
     */
    void toRGB8( uint8_t &r, uint8_t &g, uint8_t &b )
    {
	const uint8_t max = 255;
	r = this->r * max;
	g = this->g * max;
	b = this->b * max;
    }
};

/** 
 * @brief base class for an object which can be detected by the library
 */
struct Object
{
    typedef boost::shared_ptr<Object> Ptr;

    /** position of the object*/
    base::Vector3d position;
    base::Quaterniond orientation;

    /** overall likelihoof of object match **/
    double likelihood;

    /** how good does the shape match the shape model **/
    double shapeMatch;


    /** color of the object */
    Color color;

    /** @return the diameter of the sphere which encloses the object */
    virtual float boundingSphereDiameter() const = 0;

    /** @return diameter of the inner bounding sphere. This is a lower limit
     * for the objects cross-sectional shapes */
    virtual float innerSphereDiameter() const = 0;
};

struct Box : public Object
{
    /** size of the box
     * all dimensions are given in m
     */
    base::Vector3d dimensions;

    virtual float boundingSphereDiameter() const
    {
	return dimensions.norm();
    }

    virtual float innerSphereDiameter() const
    {
	return dimensions.minCoeff();
    }
};

struct Cylinder : public Object
{
    /** diameter of the cylinder in m */
    float diameter;
    /** height of the cylinder in m */
    float height;

    virtual float boundingSphereDiameter() const
    {
	return Eigen::Vector2d( diameter, height ).norm();
    }

    virtual float innerSphereDiameter() const
    {
	return std::min( diameter, height );
    }
};


/** primitive type */
    enum types 
    {
	INVALID = 0,
	BOX,
	CYLINDER
    };

/** 
 * Transport class for primitive objects, which
 * is a flat representation that is compatible with typelib
 * and can be copied.
 */
struct PrimitiveObject
{
    

    /** type of the the primitive */
    types type;

    /** parameters for the primitive types */
    base::Vector4d param;
    

    PrimitiveObject(){
       param = base::Vector4d::Zero(); 
    }

    /** 
     * construct a PrimitiveObject from an Object.
     * 
     * Effectively this is a marshaling of the primitive type hirarchie to a
     * flat structure.
     */
    explicit PrimitiveObject( const Object &o )
    {
	{
	    const Box *box = dynamic_cast<const Box*>(&o);
	    if( box )
	    {
		type = BOX;
		param.head<3>() = box->dimensions;
	    }
	}
	{
	    const Cylinder *cyl = dynamic_cast<const Cylinder*>(&o);
	    if( cyl )
	    {
		type = CYLINDER;
		param.x() = cyl->diameter;
		param.y() = cyl->height;
	    }
	}
    }

    /** 
     * return an object pointer
     */
    Object::Ptr getObject() const
    {
	Object::Ptr res;
	switch( type )
	{
	    case BOX:
	    {
		Box *box = new Box();
		box->dimensions = param.head<3>();
		res = Object::Ptr(box);
		break;
	    }
	    case CYLINDER:
	    {
		Cylinder *cyl = new Cylinder();
		cyl->diameter = param.x();
		cyl->height = param.y();
		res = Object::Ptr(cyl);
		break;
	    }
	    default:
		throw std::runtime_error( "Object type invalid" );
	}

	return res;
    }
};

};
#endif

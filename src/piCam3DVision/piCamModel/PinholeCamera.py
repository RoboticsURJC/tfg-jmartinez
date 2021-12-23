import numpy

def str2matrix(rows, cols, str):
    mat = numpy.matrix(str, dtype='float64')
    mat.resize((rows,cols))
    return mat

class Point2D:
	x = 0
	y = 0
	h = 0

class Point3D:
	x = 0
	y = 0
	z = 0
	h = 0

class PinholeCamera:
    def __init__(self, K, RT, width, height, position3D):
        # camera K y RT numpy matrix
        self.K = str2matrix(3, 3, K)
        self.RT = str2matrix(4, 4, RT)

        # 3d camera position
        self.position = position3D

        # image dimensions
        self.width = width
        self.height = height

        # camera K matrix
        self.k11 = self.K[0,0]
        self.k12 = self.K[0,1]
        self.k13 = self.K[0,2]
        self.k21 = self.K[1,0]
        self.k22 = self.K[1,1]
        self.k23 = self.K[1,2]
        self.k31 = self.K[2,0]
        self.k32 = self.K[2,1]
        self.k33 = self.K[2,2]

        # camera rotation + translation matrix 
        self.rt11 = self.RT[0,0]
        self.rt12 = self.RT[0,1]
        self.rt13 = self.RT[0,2]
        self.rt14 = self.RT[0,3]
        self.rt21 = self.RT[0,0]
        self.rt22 = self.RT[0,1]
        self.rt23 = self.RT[0,2]
        self.rt24 = self.RT[0,3]
        self.rt31 = self.RT[0,0]
        self.rt32 = self.RT[0,1]
        self.rt33 = self.RT[0,2]
        self.rt34 = self.RT[0,3]
        self.rt41 = self.RT[0,0]
        self.rt42 = self.RT[0,1]
        self.rt43 = self.RT[0,2]
        self.rt44 = self.RT[0,3]

    def backproject (self, point2D):
        output = -1
        temp2D = Point2D()
        point3D = Point3D()

        if (point2D.h>0.):
            temp2D.h=self.k11
            temp2D.x=point2D.x*self.k11/point2D.h
            temp2D.y=point2D.y*self.k11/point2D.h

            ik11=(1./self.k11)
            ik12=-self.k12/(self.k11*self.k22)
            ik13=(self.k12*self.k23-self.k13*self.k22)/(self.k22*self.k11)
            ik21=0.
            ik22=(1./self.k22)
            ik23=-(self.k23/self.k22)
            ik31=0.
            ik32=0.
            ik33=1.

            a1=ik11*temp2D.x+ik12*temp2D.y+ik13*temp2D.h
            a2=ik21*temp2D.x+ik22*temp2D.y+ik23*temp2D.h
            a3=ik31*temp2D.x+ik32*temp2D.y+ik33*temp2D.h
            a4=1.

            ir11=self.rt11
            ir12=self.rt21
            ir13=self.rt31
            ir14=0.
            ir21=self.rt12
            ir22=self.rt22
            ir23=self.rt32
            ir24=0.
            ir31=self.rt13
            ir32=self.rt23
            ir33=self.rt33
            ir34=0.
            ir41=0.
            ir42=0.
            ir43=0.
            ir44=1.

            b1=ir11*a1+ir12*a2+ir13*a3+ir14*a4
            b2=ir21*a1+ir22*a2+ir23*a3+ir24*a4
            b3=ir31*a1+ir32*a2+ir33*a3+ir34*a4
            b4=ir41*a1+ir42*a2+ir43*a3+ir44*a4 

            it11=1.
            it12=0.
            it13=0.
            it14=self.position.x
            it21=0.
            it22=1.
            it23=0.
            it24=self.position.y
            it31=0.
            it32=0.
            it33=1.
            it34=self.position.z
            it41=0.
            it42=0.
            it43=0.
            it44=1.

            point3D.x=it11*b1+it12*b2+it13*b3+it14*b4
            point3D.y=it21*b1+it22*b2+it23*b3+it24*b4
            point3D.z=it31*b1+it32*b2+it33*b3+it34*b4
            point3D.h=it41*b1+it42*b2+it43*b3+it44*b4

            if (point3D.h!=0.):
                point3D.x=point3D.x/point3D.h
                point3D.y=point3D.y/point3D.h
                point3D.z=point3D.z/point3D.h
                point3D.h=1.
                output=1
            else:
                output=0

        return(output, point3D)
        

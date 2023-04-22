import numpy as np
from matplotlib import pyplot as plt
import cv2



class meanShiftSeg:

    def __init__(self, image, windowSize):
        self.image = np.array( image, copy = True )
        assert (self.image.shape[2] == 3), "The Image must be of three channels"
        self.windowSize = 2**windowSize
        print ("Window size : " , self.windowSize)
        self.segmentedImage = np.array( image, copy = True )
        self.colorSpace = np.zeros( (256,256) )
        self.numofClusters = np.int(256/self.windowSize)**2
        print ("# of clusters : ", self.numofClusters)
        self.clustersUV = np.zeros( shape=(self.numofClusters, 2) )



    def makeColorDataSpace(self):
        
        compU = np.reshape( self.image[:,:,1], (-1,1) )
        compV = np.reshape( self.image[:,:,2], (-1,1) )
        compUV = np.transpose(np.array((compU[:,0],compV[:,0])))

        for u,v in compUV :
                # print (u, v)
                self.colorSpace[ u,v ] += 1
                
                

    def classifyColors(self):
            
            wSize = self.windowSize
            numOfWindPerDim = np.int(np.sqrt( self.numofClusters ))
            for row in range( self.image.shape[0] ):
                for col in range( self.image.shape[1] ):
                    pixelU = self.segmentedImage[row,col,1]
                    pixelV = self.segmentedImage[row,col,2]
                    windowIdx = np.int( np.int(pixelV/wSize)  + np.int(numOfWindPerDim*( pixelU/wSize )))
                    self.segmentedImage[row,col,1] = self.clustersUV[windowIdx, 0]
                    self.segmentedImage[row,col,2] = self.clustersUV[windowIdx, 1]
                    
                    
    def findCenterMass(self, window):
        
        momntIdx = range( self.windowSize )
        totalMass = np.max(np.cumsum( window ))
        if (totalMass == 0):
            return self.windowSize/2 , self.windowSize/2
        if ( totalMass > 0 ):
            #Moment around column #0 ( around the x-axis )
            momentCol = np.max(np.cumsum(window.cumsum( axis=0 )[self.windowSize-1]*momntIdx))
            cntrCol = np.round(1.0*momentCol/totalMass)
            #Moment around row #0 ( around the y-axis )
            momentRow = np.max(np.cumsum(window.cumsum( axis=1 )[:,self.windowSize-1]*momntIdx))
            cntrRow = np.round(1.0*momentRow/totalMass)

            return cntrRow, cntrCol

    
    
    
    def windowIter(self, row, col):
           
       
        wSize = self.windowSize
        hWSize = wSize/2
        prevRow = 0
        prevCol = 0
       
        window = self.colorSpace[ row:row+wSize,col:col+wSize ]
        
        newRow, newCol = self.findCenterMass( window )
        numOfIter = 0
        while( prevRow != newRow-hWSize and prevCol != newCol-hWSize ):
            if( numOfIter > np.sqrt(self.numofClusters) ):
                break

            prevRow = newCol-hWSize
            prevCol = newCol-hWSize

            nxtRow = int((prevRow+row)%(256-wSize))
            nxtCol = int((prevCol+col)%(256-wSize))
            window = self.colorSpace[ nxtRow:nxtRow+wSize,nxtCol:nxtCol+wSize ]
            newRow, newCol = self.findCenterMass( window )
            numOfIter += 1
        return row + newRow, col + newCol
    
    
    
    def applyMeanShift(self):
    
        self.makeColorDataSpace()
        wSize = self.windowSize
        
        numOfWindPerDim = np.int(np.sqrt( self.numofClusters ))
        clustersTemp = []
        for itrRow in range( numOfWindPerDim ):
            for itrCol in range( numOfWindPerDim ):
                cntrRow, cntrCol = self.windowIter(int(itrRow*wSize),int(itrCol*wSize)) 
               
                clustersTemp.append( (cntrRow, cntrCol) )

        self.clustersUV = np.array( clustersTemp )
        print (" Clusters formed ")
        print ("Clusters Centers : ")
        print (self.clustersUV)
        self.classifyColors()

        return self.segmentedImage


    
    

    


    
        
        
    
    def getSegmentedImage(self):
        return self.segmentedImage
    
    
    

if __name__== "__main__":
    path = "pic/cells.jpg"

    im = cv2.imread( path )
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    
    
    imageLUV = cv2.cvtColor( rgb, cv2.COLOR_RGB2LUV )

    meanShift = meanShiftSeg( imageLUV , 7)
    segImage = meanShift.applyMeanShift()
    
    fig , ax = plt.subplots(1,2)
    ax[0].imshow(rgb)
    ax[0].title.set_text('original_rgb')
    ax[1].imshow(segImage)
    ax[1].title.set_text('seg_img_luv')
    plt.show()        
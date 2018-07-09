import numpy as np
import urllib
import cv2

import json
import boto3

AWS_SRC_BUCKET_NAME = 'hackdaysawsmobileapp-userfiles-mobilehub-830251747'
AWS_DEST_BUCKET_NAME = 'hackdaysawsmobileapp-userfiles-mobilehub-830251747'
#AWS_DEST_BUCKET_NAME = 'hackdaysopencvexpres-processed'

def lambda_handler(event, context):
    for record in event['Records']:
        task = json.loads(record['body'])
        doTask(task)

def doTask(task):
    taskId = task['UID']
    sourceImageName = task['fileName']
    operation = task['operation']
    print("taskId=" + taskId)
    print("sourceImageName=" + sourceImageName)
    print("operation=" + operation)
    if (operation == "rotate"):
        rotateImage(taskId, sourceImageName) 
    elif (operation == "rmback"):
        removeBack(taskId, sourceImageName) 
    else:
        print("Unknown operation: " + operation)


def rotateImage(taskId, sourceImageName):
    print("Rotating " + sourceImageName)
    sourceImageUrl = "https://s3.eu-central-1.amazonaws.com/" + AWS_SRC_BUCKET_NAME + "/public/" + sourceImageName

    sourceImage = loadImage(sourceImageUrl)

    # Do transformation 
    dim = sourceImage.shape
    rotationAngle = -30
    scaleFactor = 1
    # Rotating the image by 90 degrees about the center
    # dim[0] stores the no of rows and dim[1] no of columns
    rotationMatrix = cv2.getRotationMatrix2D((dim[1] / 2, dim[0] / 2), rotationAngle, scaleFactor)
    resultImage = cv2.warpAffine(sourceImage, rotationMatrix, (dim[1], dim[0]))


    r, encodedResultImage = cv2.imencode(".png", resultImage)

    resultImagePath = "public/" + taskId + "_processed.png"
    saveToS3(bytearray(encodedResultImage), AWS_DEST_BUCKET_NAME, resultImagePath, 'image/png')

    print("Rotating result saved to " + resultImagePath)
    return resultImagePath

def removeBack(taskId, sourceImageName):
    print("Removing background " + sourceImageName)
    sourceImageUrl = "https://s3.eu-central-1.amazonaws.com/" + AWS_SRC_BUCKET_NAME + "/public/" + sourceImageName

    sourceImage = loadImage(sourceImageUrl)

    resultImage = segment(sourceImage)

    r, encodedResultImage = cv2.imencode(".png", resultImage)

    resultImagePath = "public/" + taskId + "_processed.png"
    saveToS3(bytearray(encodedResultImage), AWS_DEST_BUCKET_NAME, resultImagePath, 'image/png')

    print("Rotating result saved to " + resultImagePath)
    return resultImagePath


def loadImage(imageUrl):
    resp = urllib.urlopen(imageUrl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def saveToS3(bodyByteArray, bucketName, path, contentType):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucketName)
    bucket.put_object(
        ACL='public-read',
        ContentType=contentType,
        Key=path,
        Body=bodyByteArray
    )

def getSobel (channel):

    sobelx = cv2.Sobel(channel, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_REPLICATE)
    sobely = cv2.Sobel(channel, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_REPLICATE)
    sobel = np.hypot(sobelx, sobely)

    return sobel;

def findSignificantContours (img, sobel_8u):
    image, contours, heirarchy = cv2.findContours(sobel_8u, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find level 1 contours
    level1 = []
    for i, tupl in enumerate(heirarchy[0]):
        # Each array is in format (Next, Prev, First child, Parent)
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl, 0, [i])
            level1.append(tupl)

    # From among them, find the contours with large surface area.
    significant = []
    tooSmall = sobel_8u.size * 5 / 100 # If contour isn't covering 5% of total area of image then it probably is too small
    for tupl in level1:
        contour = contours[tupl[0]];
        area = cv2.contourArea(contour)
        if area > tooSmall:
            cv2.drawContours(img, [contour], 0, (0,255,0),2, cv2.LINE_AA, maxLevel=1)
            significant.append([contour, area])

    significant.sort(key=lambda x: x[1])
    return [x[0] for x in significant];

def segment (img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0) # Remove noise

    # Edge operator
    sobel = np.max( np.array([ getSobel(blurred[:,:, 0]), getSobel(blurred[:,:, 1]), getSobel(blurred[:,:, 2]) ]), axis=0 )

    # Noise reduction trick, from http://sourceforge.net/p/octave/image/ci/default/tree/inst/edge.m#l182
    mean = np.mean(sobel)

    # Zero any values less than mean. This reduces a lot of noise.
    sobel[sobel <= mean] = 0;
    sobel[sobel > 255] = 255;

    sobel_8u = np.asarray(sobel, np.uint8)

    # Find contours
    significant = findSignificantContours(img, sobel_8u)

    # Mask
    mask = sobel.copy()
    mask[mask > 0] = 0
    cv2.fillPoly(mask, significant, 255)
    # Invert mask
    mask = np.logical_not(mask)

    #Finally remove the background
    img[mask] = 0;
    return img

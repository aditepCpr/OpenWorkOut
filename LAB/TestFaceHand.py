class OpenWorkout:
    import json as js
    import sys
    import cv2
    import os
    from sys import platform


    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('/usr/local/python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/aditep/openpose/models"
    params["face"] = True
    params["hand"] = True

    try:
        # Starting OpenPose
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        # Process Image
        datum = op.Datum()
        imageToProcess = cv2.VideoCapture(0)

        while (True):
            # Read Video
            ret, frame = imageToProcess.read()
            # datum.openpose input data
            datum.cvInputData = frame
            opWrapper.emplaceAndPop([datum])
            print("Body keypoints: \n" + str(datum.poseKeypoints))
            print("Face keypoints: \n" + str(datum.faceKeypoints))
            print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
            print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

            try:
                # json
                keypose = str(datum.poseKeypoints)
                # key = open('keypose.json','w')
                # key = open('keypose.json','a')
                # js.dump(keypose, key, indent=4)
            except  IOError as e :
                print(e)
            cv2.imshow('frame', datum.cvOutputData)
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        imageToProcess.release()
        cv2.destroyAllWindows()




    except Exception as e:
        # print(e)
        sys.exit(-1)


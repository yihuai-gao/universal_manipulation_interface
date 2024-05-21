import cv2

def main():
    # Initialize the video capture object. 0 is the default camera
    # cap = cv2.VideoCapture("/dev/v4l/by-id/usb-Elgato_Elgato_HD60_X_A00XB32221TQU7-video-index1", cv2.CAP_V4L2)
    cap = cv2.VideoCapture(0)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    # Set properties, uncomment and adjust as necessary
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_cnt = 0
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            print(f"Read frame {frame_cnt}")
            frame_cnt += 1

            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            # Display the resulting frame
            cv2.imshow('Video Stream', frame)

            # Press 'q' on keyboard to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
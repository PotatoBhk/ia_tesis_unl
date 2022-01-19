from imutils.video import VideoStream
import cv2

user = "admin"
password = "@ezakmi1105"
ip = "192.168.1.35"
ch = 2

rtsp_url = "rtsp://{user}:{passw}@{ip}:554/h264/ch{ch}/main/av_stream"
rtsp_url = rtsp_url.format(user = user, passw = password, ip = ip, ch = ch)
salida = cv2.VideoWriter('training/videoSalida.mp4',cv2.VideoWriter_fourcc(*'mp4v'),60.0,(1280,720))

vid = VideoStream(rtsp_url).start()
while(True):
    frame = vid.read()
    cv2.imshow('video', frame)
    salida.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
      break
vid.stop()
salida.release()
cv2.destroyAllWindows()
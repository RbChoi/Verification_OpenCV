import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #손을 그리기 위한 작

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    #result 확인 위해 출력
    #print(results)
    #출력 결과는 "<class 'mediapipe.python.solution_base.SolutionOutputs'>
    #print(results.multi_hand_landmarks)
    #변화있으면 x,y,z 출력 없으면 None

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c=img.shape
                cx, cy, = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy)

                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            #읽어온 img를 손 position, 손 연결 그린다.
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    #현재 시간 읽어오
    cTime = time.time()
    #초당 프레임 수 계
    fps = 1 / (cTime - pTime)
    pTime = cTime

    #img 저장하고 fps를 string으로 변환하기
    cv2.putText(img, str(fps), (30, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)


    # ESC 키누르면 종료
    #if cv2.waitKey(1) & 0xFF == 27:
    #  break
    cap.release()
    cv2.destroyAllWindows()